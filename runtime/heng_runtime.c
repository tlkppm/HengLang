#include "heng_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <errno.h>
#include <time.h>
#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
#include <cpuid.h>
#include <immintrin.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif

enum { H_STORAGE = 100, D_F64, D_F32, D_I64, D_BOOL };
struct HObject {
    int kind, marked;
    HObject *next;
    union {
        struct { char *text; size_t size; } string;
        struct { HValue *items; size_t count, capacity; } list;
        struct { char *type; char **names; HValue *values; size_t count; } record;
        struct { int dtype; int64_t count, version; void *data; } storage;
        struct { HObject *storage; int64_t rank, count, offset; int64_t *shape, *strides; } tensor;
        struct { int csr; int64_t rows, cols, count; int64_t *row, *col; double *values; } sparse;
        struct { int ok; HValue value, error; } result;
    } as;
};
typedef union HAllocation { size_t size; max_align_t alignment; } HAllocation;
static HObject *objects;
static HFrame *frames;
static size_t live_count, used_bytes, peak_bytes, allocations, since_gc;
static void *storage_cache[32];
static size_t cache_count, cache_bytes, storage_reuses, system_allocations;
static const size_t cache_limit = 16 * 1024 * 1024;
static void trim_storage_cache(void) {
    while (cache_count) free((HAllocation *)storage_cache[--cache_count] - 1);
    cache_bytes = 0;
}
static const char *source_file = "<native>";
static int source_line, source_column;
static uint64_t rng = 88172645463325252ULL;
static int thread_count = 1, optimized = 1, matmul_fast;

void hv_site(const char *file, int line, int column) { source_file = file ? file : "<native>"; source_line = line; source_column = column; }
void hv_fail(const char *message) { fprintf(stderr, "%s:%d:%d: %s\n", source_file, source_line, source_column, message); fflush(stderr); exit(1); }
static size_t checked_size(size_t count, size_t width) { if (width && count > SIZE_MAX / width) hv_fail("分配尺寸溢出"); return count * width; }
static void *allocate(size_t size) {
    if (size > SIZE_MAX - sizeof(HAllocation) || used_bytes > SIZE_MAX - size || cache_bytes > SIZE_MAX - used_bytes - size) hv_fail("内存尺寸溢出");
    HAllocation *block = malloc(sizeof(HAllocation) + (size ? size : 1));
    if (!block) { trim_storage_cache(); block = malloc(sizeof(HAllocation) + (size ? size : 1)); }
    if (!block) hv_fail("内存不足");
    block->size = size; used_bytes += size; if (used_bytes + cache_bytes > peak_bytes) peak_bytes = used_bytes + cache_bytes;
    allocations++; system_allocations++; since_gc++; return block + 1;
}
static void release(void *pointer) { if (pointer) { HAllocation *block = (HAllocation *)pointer - 1; used_bytes -= block->size; free(block); } }
static void *allocate_storage(size_t size) {
    for (size_t index = 0; index < cache_count; index++) {
        void *pointer = storage_cache[index];
        if (((HAllocation *)pointer - 1)->size == size) {
            storage_cache[index] = storage_cache[--cache_count];
            cache_bytes -= size;
            used_bytes += size;
            allocations++;
            since_gc++;
            storage_reuses++;
            return pointer;
        }
    }
    return allocate(size);
}
static void release_storage(void *pointer) {
    if (!pointer) return;
    size_t size = ((HAllocation *)pointer - 1)->size;
    if (size >= 1024 && size <= cache_limit / 2 && cache_count < 32 && size <= cache_limit - cache_bytes) {
        used_bytes -= size;
        cache_bytes += size;
        storage_cache[cache_count++] = pointer;
        return;
    }
    release(pointer);
}
static void *resize(void *pointer, size_t size) {
    void *result = allocate(size);
    if (pointer) { size_t old = ((HAllocation *)pointer - 1)->size; memcpy(result, pointer, old < size ? old : size); release(pointer); }
    return result;
}
static char *copy_text(const char *text) { size_t length = strlen(text); if (length == SIZE_MAX) hv_fail("字符串过长"); char *result = allocate(length + 1); memcpy(result, text, length + 1); return result; }
static HValue object_value(int kind) { HObject *object = allocate(sizeof(*object)); memset(object, 0, sizeof(*object)); object->kind = kind; object->next = objects; objects = object; live_count++; HValue value = {kind, {.obj = object}}; return value; }
HValue hv_void(void) { HValue value = {H_VOID, {.i = 0}}; return value; }
HValue hv_i64(int64_t number) { HValue value = {H_I64, {.i = number}}; return value; }
HValue hv_f64(double number) { HValue value = {H_F64, {.f = number}}; return value; }
HValue hv_f32(double number) { HValue value = {H_F32, {.f = (float)number}}; return value; }
HValue hv_bool(int number) { HValue value = {H_BOOL, {.i = !!number}}; return value; }
static HObject *require(HValue value, int kind) { if (value.tag != kind || !value.as.obj || value.as.obj->kind != kind) hv_fail("值类型不匹配"); return value.as.obj; }
static int numeric(HValue value) { return value.tag == H_I64 || value.tag == H_F64 || value.tag == H_F32; }
int64_t hv_integer(HValue value) { if (value.tag != H_I64) hv_fail("需要 i64"); return value.as.i; }
double hv_number(HValue value) { if (value.tag == H_I64) return (double)value.as.i; if (value.tag == H_F64 || value.tag == H_F32) return value.as.f; hv_fail("需要数值"); return 0; }
int hv_truth(HValue value) { if (value.tag != H_BOOL) hv_fail("条件必须为 bool"); return !!value.as.i; }
HValue hv_string(const char *text) { if (!text) hv_fail("空字符串指针"); HValue value = object_value(H_STRING); value.as.obj->as.string.text = copy_text(text); value.as.obj->as.string.size = strlen(text); return value; }
HValue hv_list(int64_t count, HValue *values) {
    if (count < 0 || (count && !values)) hv_fail("列表参数无效");
    HValue value = object_value(H_LIST); HObject *object = value.as.obj;
    object->as.list.items = allocate(checked_size((size_t)count, sizeof(HValue))); object->as.list.count = object->as.list.capacity = (size_t)count;
    if (count) memcpy(object->as.list.items, values, (size_t)count * sizeof(HValue)); return value;
}
HValue hv_record(const char *type, int64_t count, const char **names, HValue *values) {
    if (!type || count < 0 || (count && (!names || !values))) hv_fail("记录参数无效");
    HValue value = object_value(H_RECORD); HObject *object = value.as.obj; object->as.record.type = copy_text(type); object->as.record.count = (size_t)count;
    object->as.record.names = allocate(checked_size((size_t)count, sizeof(char *))); object->as.record.values = allocate(checked_size((size_t)count, sizeof(HValue)));
    for (int64_t index = 0; index < count; index++) { if (!names[index]) hv_fail("字段名无效"); for (int64_t prior = 0; prior < index; prior++) if (!strcmp(names[prior], names[index])) hv_fail("重复字段名"); object->as.record.names[index] = copy_text(names[index]); object->as.record.values[index] = values[index]; } return value;
}
static size_t field_index(HObject *object, const char *name) { for (size_t index = 0; index < object->as.record.count; index++) if (!strcmp(object->as.record.names[index], name)) return index; hv_fail("记录字段不存在"); return 0; }
HValue hv_field(HValue value, const char *name) { HObject *object = require(value, H_RECORD); return object->as.record.values[field_index(object, name)]; }
void hv_set_field(HValue value, const char *name, HValue item) { HObject *object = require(value, H_RECORD); object->as.record.values[field_index(object, name)] = item; }
int64_t hv_length(HValue value) { if (value.tag == H_LIST) return (int64_t)require(value,H_LIST)->as.list.count; if (value.tag == H_STRING) return (int64_t)require(value,H_STRING)->as.string.size; if (value.tag == H_TENSOR) return require(value,H_TENSOR)->as.tensor.count; hv_fail("该类型不支持 len"); return 0; }
HValue hv_index(HValue value, HValue index) { HObject *list = require(value, H_LIST); int64_t position = hv_integer(index); if (position < 0 || (uint64_t)position >= list->as.list.count) hv_fail("列表索引越界"); return list->as.list.items[position]; }
void hv_set_index(HValue value, HValue index, HValue item) { HObject *list = require(value,H_LIST); int64_t position = hv_integer(index); if (position < 0 || (uint64_t)position >= list->as.list.count) hv_fail("列表索引越界"); list->as.list.items[position] = item; }
void hv_frame_push(HFrame *frame, HValue *slots, size_t count) { if (!frame || (count && !slots)) hv_fail("根槽无效"); frame->slots = slots; frame->count = count; frame->previous = frames; frames = frame; }
void hv_frame_pop(HFrame *frame) { if (frames != frame) hv_fail("根帧顺序错误"); frames = frame->previous; }
static void destroy(HObject *object) {
    switch (object->kind) {
        case H_STRING: release(object->as.string.text); break;
        case H_LIST: release(object->as.list.items); break;
        case H_RECORD: for (size_t index = 0; index < object->as.record.count; index++) release(object->as.record.names[index]); release(object->as.record.names); release(object->as.record.values); release(object->as.record.type); break;
        case H_STORAGE: release_storage(object->as.storage.data); break;
        case H_TENSOR: release(object->as.tensor.shape); release(object->as.tensor.strides); break;
        case H_SPARSE: release(object->as.sparse.row); release(object->as.sparse.col); release(object->as.sparse.values); break;
    }
    release(object); live_count--;
}
static void enqueue(HValue value, HObject **work, size_t *count) { if (value.tag >= H_STRING && value.as.obj && !value.as.obj->marked) { value.as.obj->marked = 1; work[(*count)++] = value.as.obj; } }
static void collect(void) {
    HObject **work = malloc(checked_size(live_count ? live_count : 1, sizeof(HObject *))); if (!work) hv_fail("GC 内存不足"); size_t count = 0;
    for (HFrame *frame = frames; frame; frame = frame->previous) for (size_t index = 0; index < frame->count; index++) enqueue(frame->slots[index],work,&count);
    while (count) { HObject *object = work[--count]; switch (object->kind) {
        case H_LIST: for (size_t index=0;index<object->as.list.count;index++) enqueue(object->as.list.items[index],work,&count); break;
        case H_RECORD: for (size_t index=0;index<object->as.record.count;index++) enqueue(object->as.record.values[index],work,&count); break;
        case H_TENSOR: { HValue storage = {H_STORAGE,{.obj=object->as.tensor.storage}}; enqueue(storage,work,&count); break; }
        case H_RESULT: enqueue(object->as.result.value,work,&count); enqueue(object->as.result.error,work,&count); break;
    }}
    free(work); HObject **link = &objects;
    while (*link) { HObject *object = *link; if (object->marked) { object->marked = 0; link = &object->next; } else { *link = object->next; destroy(object); } } since_gc = 0;
}
void hv_safepoint(void) { if (since_gc >= 256) collect(); }
void hv_shutdown(void) { frames = NULL; collect(); trim_storage_cache(); }
static int64_t add_i64(int64_t left,int64_t right) { if ((right>0 && left>INT64_MAX-right)||(right<0 && left<INT64_MIN-right)) hv_fail("i64 加法溢出"); return left+right; }
static int64_t sub_i64(int64_t left,int64_t right) { if ((right>0 && left<INT64_MIN+right)||(right<0 && left>INT64_MAX+right)) hv_fail("i64 减法溢出"); return left-right; }
static int64_t mul_i64(int64_t left,int64_t right) {
    if (left && right) { if (left > 0 ? (right > 0 ? left > INT64_MAX/right : right < INT64_MIN/left) : (right > 0 ? left < INT64_MIN/right : left < INT64_MAX/right)) hv_fail("i64 乘法溢出"); } return left*right;
}
static int compare_numeric(HValue left,HValue right){
    if(left.tag==H_I64&&right.tag==H_I64)return left.as.i<right.as.i?-1:left.as.i>right.as.i?1:0;
    if(left.tag==H_I64){double number=hv_number(right);if(isnan(number))return 2;if(number>=9223372036854775808.0)return -1;if(number<-9223372036854775808.0)return 1;int64_t integer=(int64_t)number;if(left.as.i<integer)return -1;if(left.as.i>integer)return 1;return (double)integer<number?-1:(double)integer>number?1:0;}
    if(right.tag==H_I64){int result=compare_numeric(right,left);return result==2?2:-result;}
    double a=hv_number(left),b=hv_number(right);if(isnan(a)||isnan(b))return 2;return a<b?-1:a>b?1:0;
}
static int equal_value(HValue left,HValue right) {
    if (numeric(left)&&numeric(right)) return compare_numeric(left,right)==0;
    if (left.tag!=right.tag) return 0; if (left.tag==H_VOID) return 1; if (left.tag==H_BOOL) return left.as.i==right.as.i; if (left.tag==H_STRING) return !strcmp(left.as.obj->as.string.text,right.as.obj->as.string.text); return left.as.obj==right.as.obj;
}
static HValue binary_tensor_values(const char *operation,HValue left,HValue right);
static HValue unary_tensor_value(const char *operation,HValue value);
HValue hv_binary(const char *operation,HValue left,HValue right) {
    if(left.tag==H_TENSOR||right.tag==H_TENSOR)return binary_tensor_values(operation,left,right);
    if (!strcmp(operation,"==")) return hv_bool(equal_value(left,right)); if (!strcmp(operation,"!=")) return hv_bool(!equal_value(left,right));
    if (!strcmp(operation,"and")||!strcmp(operation,"&&")) return hv_bool(hv_truth(left)&hv_truth(right)); if (!strcmp(operation,"or")||!strcmp(operation,"||")) return hv_bool(hv_truth(left)|hv_truth(right));
    if (!strcmp(operation,"+")&&left.tag==H_STRING&&right.tag==H_STRING) { size_t first=left.as.obj->as.string.size,second=right.as.obj->as.string.size; if (first>SIZE_MAX-second-1) hv_fail("字符串尺寸溢出"); char *text=allocate(first+second+1); memcpy(text,left.as.obj->as.string.text,first); memcpy(text+first,right.as.obj->as.string.text,second+1); HValue result=hv_string(text); release(text); return result; }
    if (!numeric(left)||!numeric(right)) hv_fail("运算需要数值");
    if (left.tag==H_I64&&right.tag==H_I64) { int64_t a=left.as.i,b=right.as.i;
        if (!strcmp(operation,"+")) return hv_i64(add_i64(a,b)); if (!strcmp(operation,"-")) return hv_i64(sub_i64(a,b)); if (!strcmp(operation,"*")) return hv_i64(mul_i64(a,b));
        if (!strcmp(operation,"/")||!strcmp(operation,"%")||!strcmp(operation,"//")) { if (!b) hv_fail("整数除零"); if (a==INT64_MIN&&b==-1) hv_fail("整数除法溢出"); return hv_i64(!strcmp(operation,"%")?a%b:a/b); }
        if (!strcmp(operation,"<")) return hv_bool(a<b); if (!strcmp(operation,">")) return hv_bool(a>b); if (!strcmp(operation,"<=")) return hv_bool(a<=b); if (!strcmp(operation,">=")) return hv_bool(a>=b);
    }
    int comparison=compare_numeric(left,right);
    if (!strcmp(operation,"<")) return hv_bool(comparison==-1); if (!strcmp(operation,">")) return hv_bool(comparison==1); if (!strcmp(operation,"<=")) return hv_bool(comparison==-1||comparison==0); if (!strcmp(operation,">=")) return hv_bool(comparison==1||comparison==0);
    double a=hv_number(left),b=hv_number(right),result;
    if (!strcmp(operation,"+")) result=a+b; else if (!strcmp(operation,"-")) result=a-b; else if (!strcmp(operation,"*")) result=a*b; else if (!strcmp(operation,"/")) { if (b==0) hv_fail("除零"); result=a/b; } else if (!strcmp(operation,"%")) { if(b==0) hv_fail("除零"); result=fmod(a,b); } else if (!strcmp(operation,"**")) result=pow(a,b); else { hv_fail("未知二元运算"); return hv_void(); }
    return left.tag==H_F32&&right.tag==H_F32?hv_f32(result):hv_f64(result);
}
HValue hv_unary(const char *operation,HValue value) {if(value.tag==H_TENSOR)return unary_tensor_value(operation,value); if (!strcmp(operation,"!")||!strcmp(operation,"not")) return hv_bool(!hv_truth(value)); if (!strcmp(operation,"+")) { hv_number(value); return value; } if (!strcmp(operation,"-")) { if (value.tag==H_I64) { if(value.as.i==INT64_MIN) hv_fail("整数取负溢出"); return hv_i64(-value.as.i); } return value.tag==H_F32?hv_f32(-hv_number(value)):hv_f64(-hv_number(value)); } hv_fail("未知一元运算"); return hv_void(); }
static const char *text_of(HValue value) { return require(value,H_STRING)->as.string.text; }
static int64_t list_int(HValue list,int64_t index) { return hv_integer(hv_index(list,hv_i64(index))); }
static size_t dtype_size(int dtype) { switch(dtype) { case D_F64:return sizeof(double); case D_F32:return sizeof(float); case D_I64:return sizeof(int64_t); case D_BOOL:return sizeof(unsigned char); } hv_fail("无效 dtype"); return 0; }
static const char *dtype_name(int dtype) { switch(dtype) { case D_F64:return "f64"; case D_F32:return "f32"; case D_I64:return "i64"; case D_BOOL:return "bool"; } hv_fail("无效 dtype"); return ""; }
static int parse_dtype(const char *name) { if(!strcmp(name,"f64"))return D_F64; if(!strcmp(name,"f32"))return D_F32; if(!strcmp(name,"i64"))return D_I64; if(!strcmp(name,"bool"))return D_BOOL; hv_fail("未知 dtype");return 0; }
static int64_t shape_count(int64_t rank,const int64_t *shape) { int empty=0; for(int64_t axis=0;axis<rank;axis++) { if(shape[axis]<0)hv_fail("负张量维度"); if(!shape[axis])empty=1; } if(empty)return 0;int64_t count=1; for(int64_t axis=0;axis<rank;axis++) count=mul_i64(count,shape[axis]); return count; }
static HValue tensor_allocate(int64_t rank,const int64_t *shape,int dtype,int zero){
    if(rank<0)hv_fail("张量 rank 不能为负"); int64_t count=shape_count(rank,shape);
    HValue value=object_value(H_TENSOR); HObject *tensor=value.as.obj; tensor->as.tensor.rank=rank;tensor->as.tensor.count=count;
    tensor->as.tensor.shape=allocate(checked_size((size_t)rank,sizeof(int64_t)));tensor->as.tensor.strides=allocate(checked_size((size_t)rank,sizeof(int64_t)));
    int64_t stride=count?1:0; for(int64_t axis=rank;axis-->0;) { tensor->as.tensor.shape[axis]=shape[axis];tensor->as.tensor.strides[axis]=stride;stride=mul_i64(stride,shape[axis]); }
    HValue storage=object_value(H_STORAGE);tensor->as.tensor.storage=storage.as.obj;storage.as.obj->as.storage.dtype=dtype;storage.as.obj->as.storage.count=count;
    size_t bytes=checked_size((size_t)count,dtype_size(dtype));storage.as.obj->as.storage.data=allocate_storage(bytes);
    if(zero)memset(storage.as.obj->as.storage.data,0,bytes);
    return value;
}
static HValue tensor_new(int64_t rank,const int64_t *shape,int dtype){return tensor_allocate(rank,shape,dtype,1);}
static HValue tensor_from_shape(HValue shape,int dtype) {require(shape,H_LIST);int64_t rank=hv_length(shape);int64_t *dimensions=allocate(checked_size((size_t)rank,sizeof(int64_t)));for(int64_t axis=0;axis<rank;axis++)dimensions[axis]=list_int(shape,axis);HValue result=tensor_new(rank,dimensions,dtype);release(dimensions);return result;}
static int tensor_dtype(HObject *tensor) { return tensor->as.tensor.storage->as.storage.dtype; }
static int64_t tensor_offset(HObject *tensor,int64_t index) { if(index<0||index>=tensor->as.tensor.count)hv_fail("张量索引越界");int64_t offset=tensor->as.tensor.offset;for(int64_t axis=tensor->as.tensor.rank;axis-->0;) { int64_t size=tensor->as.tensor.shape[axis];offset=add_i64(offset,mul_i64(index%size,tensor->as.tensor.strides[axis]));index/=size; } if(offset<0||offset>=tensor->as.tensor.storage->as.storage.count)hv_fail("张量 view 越界");return offset; }
static double read_offset(HObject *tensor,int64_t offset) { void *data=tensor->as.tensor.storage->as.storage.data;switch(tensor_dtype(tensor)) {case D_F64:return ((double*)data)[offset];case D_F32:return ((float*)data)[offset];case D_I64:return (double)((int64_t*)data)[offset];case D_BOOL:return ((unsigned char*)data)[offset];}return 0; }
static HValue numeric_offset(HObject *tensor,int64_t offset){if(tensor_dtype(tensor)==D_I64)return hv_i64(((int64_t*)tensor->as.tensor.storage->as.storage.data)[offset]);return hv_f64(read_offset(tensor,offset));}
static double tensor_read(HObject *tensor,int64_t index) { return read_offset(tensor,tensor_offset(tensor,index)); }
static int64_t number_to_int(double value) { if(!isfinite(value)||value<-9223372036854775808.0||value>=9223372036854775808.0)hv_fail("数值超出 i64 范围");return (int64_t)value; }
static void write_offset(HObject *tensor,int64_t offset,double value) {void *data=tensor->as.tensor.storage->as.storage.data;switch(tensor_dtype(tensor)){case D_F64:((double*)data)[offset]=value;break;case D_F32:((float*)data)[offset]=(float)value;break;case D_I64:((int64_t*)data)[offset]=number_to_int(value);break;case D_BOOL:((unsigned char*)data)[offset]=(value!=0);break;}}
static void tensor_write(HObject *tensor,int64_t index,double value) {write_offset(tensor,tensor_offset(tensor,index),value);}
static void copy_element(HObject *destination,int64_t to,HObject *source,int64_t from) {int64_t target=tensor_offset(destination,to),origin=tensor_offset(source,from);if(tensor_dtype(destination)==tensor_dtype(source)){size_t width=dtype_size(tensor_dtype(source));memcpy((char*)destination->as.tensor.storage->as.storage.data+(size_t)target*width,(char*)source->as.tensor.storage->as.storage.data+(size_t)origin*width,width);}else write_offset(destination,target,read_offset(source,origin));}
static int tensor_contiguous(HObject *tensor);
static HValue tensor_clone(HObject *source) {
    HValue result=tensor_allocate(source->as.tensor.rank,source->as.tensor.shape,tensor_dtype(source),0);
    int64_t count=source->as.tensor.count;
    if(!count)return result;
    if(tensor_contiguous(source)){
        size_t width=dtype_size(tensor_dtype(source));
        const char *data=source->as.tensor.storage->as.storage.data;
        memcpy(result.as.obj->as.tensor.storage->as.storage.data,data+(size_t)source->as.tensor.offset*width,(size_t)count*width);
        return result;
    }
    for(int64_t index=0;index<count;index++)copy_element(result.as.obj,index,source,index);
    return result;
}
static HValue tensor_view(HObject *source) {HValue result=object_value(H_TENSOR);HObject *view=result.as.obj;view->as.tensor=source->as.tensor;size_t bytes=checked_size((size_t)source->as.tensor.rank,sizeof(int64_t));view->as.tensor.shape=allocate(bytes);view->as.tensor.strides=allocate(bytes);memcpy(view->as.tensor.shape,source->as.tensor.shape,bytes);memcpy(view->as.tensor.strides,source->as.tensor.strides,bytes);return result;}
static int64_t axis_value(HObject *tensor,int64_t axis) {if(axis<0)axis+=tensor->as.tensor.rank;if(axis<0||axis>=tensor->as.tensor.rank)hv_fail("axis 越界");return axis;}
static int64_t indexed_offset(HObject *tensor,HValue indices) {require(indices,H_LIST);if(hv_length(indices)!=tensor->as.tensor.rank)hv_fail("索引 rank 不匹配");int64_t offset=tensor->as.tensor.offset;for(int64_t axis=0;axis<tensor->as.tensor.rank;axis++){int64_t index=list_int(indices,axis);if(index<0||index>=tensor->as.tensor.shape[axis])hv_fail("张量索引越界");offset=add_i64(offset,mul_i64(index,tensor->as.tensor.strides[axis]));}return offset;}
static int promoted(int left,int right) {if(left==D_F64||right==D_F64)return D_F64;if(left==D_F32||right==D_F32)return D_F32;if(left==D_I64||right==D_I64)return D_I64;return D_BOOL;}
static int64_t broadcast_shape(HObject *left,HObject *right,int64_t *shape) {int64_t rank=left->as.tensor.rank>right->as.tensor.rank?left->as.tensor.rank:right->as.tensor.rank;for(int64_t axis=0;axis<rank;axis++){int64_t li=axis-(rank-left->as.tensor.rank),ri=axis-(rank-right->as.tensor.rank),a=li<0?1:left->as.tensor.shape[li],b=ri<0?1:right->as.tensor.shape[ri];if(a!=b&&a!=1&&b!=1)hv_fail("广播 shape 不兼容");shape[axis]=a==1?b:a;}return rank;}
static int64_t broadcast_offset(HObject *tensor,int64_t rank,const int64_t *shape,int64_t index) {int64_t offset=tensor->as.tensor.offset;for(int64_t axis=rank;axis-->0;){int64_t coordinate=index%shape[axis];index/=shape[axis];int64_t own=axis-(rank-tensor->as.tensor.rank);if(own>=0&&tensor->as.tensor.shape[own]!=1)offset=add_i64(offset,mul_i64(coordinate,tensor->as.tensor.strides[own]));}return offset;}
static int tensor_contiguous(HObject *tensor){
    if(!tensor->as.tensor.count)return 1;
    int64_t stride=1;
    for(int64_t axis=tensor->as.tensor.rank;axis-->0;){
        if(tensor->as.tensor.shape[axis]>1&&tensor->as.tensor.strides[axis]!=stride)return 0;
        stride=mul_i64(stride,tensor->as.tensor.shape[axis]);
    }
    return 1;
}
#define FLOAT_BINARY_LOOP(expression) do { for(int64_t index=0;index<count;index++){double a=first[index*left_step],b=second[index*right_step];target[index]=(expression);} } while(0)
#define FLOAT_BINARY_TYPES(left_type,right_type,output_type) do { \
    const left_type *first=(const left_type*)left->as.tensor.storage->as.storage.data+left->as.tensor.offset; \
    const right_type *second=(const right_type*)right->as.tensor.storage->as.storage.data+right->as.tensor.offset; \
    output_type *target=output->as.tensor.storage->as.storage.data; \
    if(!strcmp(name,"add"))FLOAT_BINARY_LOOP(a+b); \
    else if(!strcmp(name,"sub"))FLOAT_BINARY_LOOP(a-b); \
    else if(!strcmp(name,"mul"))FLOAT_BINARY_LOOP(a*b); \
    else {for(int64_t index=0;index<count;index++){double a=first[index*left_step],b=second[index*right_step];if(b==0)hv_fail("张量除零");target[index]=a/b;}} \
} while(0)
#define FLOAT_MATRIX_BINARY(expression) do { \
    for(int64_t row=0;row<rows;row++){ \
        const double *first_row=first+row*left_row,*second_row=second+row*right_row; \
        for(int64_t column=0;column<columns;column++){ \
            double a=first_row[column*left_column],b=second_row[column*right_column]; \
            target[row*columns+column]=(expression); \
        } \
    } \
} while(0)
static int tensor_binary_matrix(const char *name,HObject *left,HObject *right,HObject *output){
    if(output->as.tensor.rank!=2)return 0;
    const int64_t *left_shape=left->as.tensor.shape,*right_shape=right->as.tensor.shape;
    const int64_t *left_strides=left->as.tensor.strides,*right_strides=right->as.tensor.strides;
    int64_t lr=left->as.tensor.rank,rr=right->as.tensor.rank;
    int64_t left_column=lr&&left_shape[lr-1]!=1?left_strides[lr-1]:0;
    int64_t right_column=rr&&right_shape[rr-1]!=1?right_strides[rr-1]:0;
    if(left_column<0||left_column>1||right_column<0||right_column>1)return 0;
    int64_t left_row=lr==2&&left_shape[0]!=1?left_strides[0]:0;
    int64_t right_row=rr==2&&right_shape[0]!=1?right_strides[0]:0;
    const double *first=(const double*)left->as.tensor.storage->as.storage.data+left->as.tensor.offset;
    const double *second=(const double*)right->as.tensor.storage->as.storage.data+right->as.tensor.offset;
    double *target=output->as.tensor.storage->as.storage.data;
    int64_t rows=output->as.tensor.shape[0],columns=output->as.tensor.shape[1];
    if(!strcmp(name,"add"))FLOAT_MATRIX_BINARY(a+b);
    else if(!strcmp(name,"sub"))FLOAT_MATRIX_BINARY(a-b);
    else if(!strcmp(name,"mul"))FLOAT_MATRIX_BINARY(a*b);
    else return 0;
    return 1;
}
#undef FLOAT_MATRIX_BINARY
static int tensor_binary_float(const char *name,HObject *left,HObject *right,HObject *output){
    int ld=tensor_dtype(left),rd=tensor_dtype(right);
    if((ld!=D_F64&&ld!=D_F32)||(rd!=D_F64&&rd!=D_F32)||!strcmp(name,"equal"))return 0;
    int64_t count=output->as.tensor.count;
    if(!count)return 1;
    if((left->as.tensor.count!=count&&left->as.tensor.count!=1)||(right->as.tensor.count!=count&&right->as.tensor.count!=1)||!tensor_contiguous(left)||!tensor_contiguous(right))
        return ld==D_F64&&rd==D_F64&&tensor_binary_matrix(name,left,right,output);
    int64_t left_step=left->as.tensor.count==1?0:1,right_step=right->as.tensor.count==1?0:1;
    if(ld==D_F64){if(rd==D_F64)FLOAT_BINARY_TYPES(double,double,double);else FLOAT_BINARY_TYPES(double,float,double);}
    else {if(rd==D_F64)FLOAT_BINARY_TYPES(float,double,double);else FLOAT_BINARY_TYPES(float,float,float);}
    return 1;
}
#undef FLOAT_BINARY_TYPES
#undef FLOAT_BINARY_LOOP
static HValue tensor_binary(const char *name,HObject *left,HObject *right){int64_t capacity=left->as.tensor.rank>right->as.tensor.rank?left->as.tensor.rank:right->as.tensor.rank;int64_t *shape=allocate(checked_size((size_t)capacity,sizeof(int64_t)));int64_t rank=broadcast_shape(left,right,shape);int dtype=!strcmp(name,"equal")?D_BOOL:promoted(tensor_dtype(left),tensor_dtype(right));if(dtype==D_BOOL&&strcmp(name,"equal"))dtype=D_I64;if(!strcmp(name,"div")&&dtype==D_I64)dtype=D_F64;HValue result=tensor_new(rank,shape,dtype);release(shape);if(tensor_binary_float(name,left,right,result.as.obj))return result;shape=result.as.obj->as.tensor.shape;for(int64_t index=0;index<result.as.obj->as.tensor.count;index++){int64_t lo=broadcast_offset(left,rank,shape,index),ro=broadcast_offset(right,rank,shape,index);if((dtype==D_I64||!strcmp(name,"equal"))&&(tensor_dtype(left)==D_I64||tensor_dtype(left)==D_BOOL)&&(tensor_dtype(right)==D_I64||tensor_dtype(right)==D_BOOL)){int64_t a=tensor_dtype(left)==D_I64?((int64_t*)left->as.tensor.storage->as.storage.data)[lo]:((unsigned char*)left->as.tensor.storage->as.storage.data)[lo],b=tensor_dtype(right)==D_I64?((int64_t*)right->as.tensor.storage->as.storage.data)[ro]:((unsigned char*)right->as.tensor.storage->as.storage.data)[ro];if(!strcmp(name,"equal"))tensor_write(result.as.obj,index,a==b);else ((int64_t*)result.as.obj->as.tensor.storage->as.storage.data)[index]=!strcmp(name,"add")?add_i64(a,b):!strcmp(name,"sub")?sub_i64(a,b):mul_i64(a,b);continue;}double a=read_offset(left,lo),b=read_offset(right,ro),value;if(!strcmp(name,"add"))value=a+b;else if(!strcmp(name,"sub"))value=a-b;else if(!strcmp(name,"mul"))value=a*b;else if(!strcmp(name,"div")){if(b==0)hv_fail("张量除零");value=a/b;}else value=compare_numeric(numeric_offset(left,lo),numeric_offset(right,ro))==0;tensor_write(result.as.obj,index,value);}return result;}
static HValue scalar_tensor(HValue value){if(value.tag==H_TENSOR)return value;if(!numeric(value))hv_fail("张量运算仅支持张量或数值");int dtype=value.tag==H_I64?D_I64:value.tag==H_F32?D_F32:D_F64;HValue result=tensor_new(0,NULL,dtype);if(dtype==D_I64)((int64_t*)result.as.obj->as.tensor.storage->as.storage.data)[0]=value.as.i;else tensor_write(result.as.obj,0,value.as.f);return result;}
static HValue binary_tensor_values(const char *operation,HValue left,HValue right){const char *name;if(!strcmp(operation,"+"))name="add";else if(!strcmp(operation,"-"))name="sub";else if(!strcmp(operation,"*"))name="mul";else if(!strcmp(operation,"/"))name="div";else{hv_fail("张量不支持该运算符");return hv_void();}left=scalar_tensor(left);right=scalar_tensor(right);return tensor_binary(name,left.as.obj,right.as.obj);}
static HValue unary_tensor_value(const char *operation,HValue value){if(!strcmp(operation,"+"))return value;if(strcmp(operation,"-"))hv_fail("张量不支持该一元运算符");HObject *source=require(value,H_TENSOR);int dtype=tensor_dtype(source)==D_BOOL?D_I64:tensor_dtype(source);HValue result=tensor_new(source->as.tensor.rank,source->as.tensor.shape,dtype);for(int64_t index=0;index<source->as.tensor.count;index++){if(dtype==D_I64){int64_t number=tensor_dtype(source)==D_I64?((int64_t*)source->as.tensor.storage->as.storage.data)[tensor_offset(source,index)]:(int64_t)tensor_read(source,index);if(number==INT64_MIN)hv_fail("整数张量取负溢出");((int64_t*)result.as.obj->as.tensor.storage->as.storage.data)[index]=-number;}else tensor_write(result.as.obj,index,-tensor_read(source,index));}return result;}
static uint64_t random_bits(void){uint64_t value=rng;value^=value>>12;value^=value<<25;value^=value>>27;rng=value;return value*2685821657736338717ULL;}
static double random_unit(void){return (double)(random_bits()>>11)*(1.0/9007199254740992.0);}
static double sum_f64(const double *values,int64_t count){
    double total=0,correction=0;
    for(int64_t index=0;index<count;index++){
        double adjusted=values[index]-correction,next=total+adjusted;
        correction=isfinite(next)?(next-total)-adjusted:0;
        total=next;
    }
    return total;
}
static int tensor_reduce_last_f64(HObject *source,HObject *output,int64_t axis,int arg,int mean,int maximum){
    if(tensor_dtype(source)!=D_F64||axis!=source->as.tensor.rank-1||!tensor_contiguous(source))return 0;
    int64_t width=source->as.tensor.shape[axis];
    if(!width||!source->as.tensor.count)return 1;
    const double *values=(const double*)source->as.tensor.storage->as.storage.data+source->as.tensor.offset;
    for(int64_t index=0;index<output->as.tensor.count;index++){
        const double *row=values+index*width;
        if(arg||maximum){
            int64_t best=0;
            for(int64_t column=1;column<width;column++)if(row[column]>row[best]||isnan(row[column]))best=column;
            if(arg)((int64_t*)output->as.tensor.storage->as.storage.data)[index]=best;
            else ((double*)output->as.tensor.storage->as.storage.data)[index]=row[best];
        }else{
            double total=sum_f64(row,width);
            ((double*)output->as.tensor.storage->as.storage.data)[index]=mean?total/(double)width:total;
        }
    }
    return 1;
}
static HValue tensor_reduce(const char *name,HObject *source,int64_t axis,int keep){
    axis=axis_value(source,axis);int arg=!strcmp(name,"argmax"),mean=!strcmp(name,"mean"),maximum=!strcmp(name,"max");int64_t rank=0;int64_t *shape=allocate(checked_size((size_t)source->as.tensor.rank,sizeof(int64_t)));
    for(int64_t dimension=0;dimension<source->as.tensor.rank;dimension++)if(dimension!=axis)shape[rank++]=source->as.tensor.shape[dimension];else if(keep)shape[rank++]=1;
    int dtype=arg?D_I64:mean&&(tensor_dtype(source)==D_I64||tensor_dtype(source)==D_BOOL)?D_F64:tensor_dtype(source)==D_BOOL&&!maximum?D_I64:tensor_dtype(source);
    HValue result=tensor_new(rank,shape,dtype);release(shape);int64_t width=source->as.tensor.shape[axis];if(!width&&(mean||maximum||arg)&&result.as.obj->as.tensor.count)hv_fail("空轴无法执行该归约");
    if(tensor_reduce_last_f64(source,result.as.obj,axis,arg,mean,maximum))return result;
    for(int64_t index=0;index<result.as.obj->as.tensor.count;index++){
        int64_t remainder=index,offset=source->as.tensor.offset;for(int64_t dimension=source->as.tensor.rank;dimension-->0;){if(dimension==axis)continue;int64_t coordinate=remainder%source->as.tensor.shape[dimension];remainder/=source->as.tensor.shape[dimension];offset=add_i64(offset,mul_i64(coordinate,source->as.tensor.strides[dimension]));}
        double total=0,correction=0,best=-INFINITY;int64_t best_index=0,integer_total=0,integer_best=INT64_MIN;
        for(int64_t item=0;item<width;item++){int64_t origin=add_i64(offset,mul_i64(item,source->as.tensor.strides[axis]));double value=read_offset(source,origin);
            if(maximum||arg){if(tensor_dtype(source)==D_I64){int64_t exact=((int64_t*)source->as.tensor.storage->as.storage.data)[origin];if(item==0||exact>integer_best){integer_best=exact;best_index=item;}}else if(item==0||value>best||isnan(value)){best=value;best_index=item;}}
            else if(dtype==D_I64){int64_t exact=tensor_dtype(source)==D_I64?((int64_t*)source->as.tensor.storage->as.storage.data)[origin]:(int64_t)value;integer_total=add_i64(integer_total,exact);}
            else {double adjusted=value-correction,next=total+adjusted;correction=isfinite(next)?(next-total)-adjusted:0;total=next;}
        }
        if(arg)((int64_t*)result.as.obj->as.tensor.storage->as.storage.data)[index]=best_index;
        else if(dtype==D_I64)((int64_t*)result.as.obj->as.tensor.storage->as.storage.data)[index]=maximum?integer_best:integer_total;
        else tensor_write(result.as.obj,index,maximum?best:mean?total/(double)width:total);
    }return result;
}
static HValue tensor_sum_all(HObject *source){int dtype=tensor_dtype(source)==D_BOOL?D_I64:tensor_dtype(source);HValue result=tensor_new(0,NULL,dtype);if(dtype==D_F64&&tensor_contiguous(source)){if(source->as.tensor.count){const double *values=(const double*)source->as.tensor.storage->as.storage.data+source->as.tensor.offset;((double*)result.as.obj->as.tensor.storage->as.storage.data)[0]=sum_f64(values,source->as.tensor.count);}return result;}double total=0,correction=0;int64_t exact=0;for(int64_t index=0;index<source->as.tensor.count;index++){if(dtype==D_I64){int64_t value=tensor_dtype(source)==D_I64?((int64_t*)source->as.tensor.storage->as.storage.data)[tensor_offset(source,index)]:(int64_t)tensor_read(source,index);exact=add_i64(exact,value);}else{double adjusted=tensor_read(source,index)-correction,next=total+adjusted;correction=isfinite(next)?(next-total)-adjusted:0;total=next;}}if(dtype==D_I64)((int64_t*)result.as.obj->as.tensor.storage->as.storage.data)[0]=exact;else tensor_write(result.as.obj,0,total);return result;}
static HValue tensor_reshape(HObject *source,HValue target){
    require(target,H_LIST);int64_t rank=hv_length(target),unknown=-1,known=1;int64_t *shape=allocate(checked_size((size_t)rank,sizeof(int64_t)));
    for(int64_t axis=0;axis<rank;axis++){shape[axis]=list_int(target,axis);if(shape[axis]==-1){if(unknown>=0)hv_fail("reshape 仅允许一个推断维度");unknown=axis;shape[axis]=1;}else if(shape[axis]<0)hv_fail("reshape 维度无效");}
    known=shape_count(rank,shape);if(unknown>=0){if(!known||source->as.tensor.count%known)hv_fail("reshape 无法推断维度");shape[unknown]=source->as.tensor.count/known;}
    if(shape_count(rank,shape)!=source->as.tensor.count)hv_fail("reshape 元素数量不匹配");
    int contiguous=1;int64_t stride=source->as.tensor.count?1:0;for(int64_t axis=source->as.tensor.rank;axis-->0;){if(source->as.tensor.count&&source->as.tensor.shape[axis]>1&&source->as.tensor.strides[axis]!=stride)contiguous=0;stride=mul_i64(stride,source->as.tensor.shape[axis]);}
    HObject *base=source;if(!contiguous)base=tensor_clone(source).as.obj;HValue result=tensor_view(base);HObject *view=result.as.obj;release(view->as.tensor.shape);release(view->as.tensor.strides);view->as.tensor.shape=shape;view->as.tensor.strides=allocate(checked_size((size_t)rank,sizeof(int64_t)));view->as.tensor.rank=rank;stride=source->as.tensor.count?1:0;for(int64_t axis=rank;axis-->0;){view->as.tensor.strides[axis]=stride;stride=mul_i64(stride,shape[axis]);}return result;
}
static HValue tensor_transpose(HObject *source,HValue axes){require(axes,H_LIST);int64_t rank=source->as.tensor.rank;if(hv_length(axes)!=rank)hv_fail("transpose axes 数量不匹配");HValue result=tensor_view(source);unsigned char *seen=allocate((size_t)rank);memset(seen,0,(size_t)rank);for(int64_t index=0;index<rank;index++){int64_t axis=axis_value(source,list_int(axes,index));if(seen[axis])hv_fail("transpose axes 重复");seen[axis]=1;result.as.obj->as.tensor.shape[index]=source->as.tensor.shape[axis];result.as.obj->as.tensor.strides[index]=source->as.tensor.strides[axis];}release(seen);return result;}
static HValue tensor_slice(HObject *source,int64_t axis,int64_t start,int64_t end,int64_t step){
    axis=axis_value(source,axis);int64_t size=source->as.tensor.shape[axis],length=0;if(!step||step==INT64_MIN)hv_fail("slice 步长无效");
    if(step>0){if(start<0)start=add_i64(start,size);if(end<0)end=add_i64(end,size);if(start<0||start>size||end<0||end>size)hv_fail("slice 区间越界");if(end>start)length=1+(end-start-1)/step;}
    else{if(start<0)start=add_i64(start,size);if(end<-1)end=add_i64(end,size);if(size==0){if(start!=0||end!=-1)hv_fail("slice 空区间越界");}else if(start<0||start>=size||end<-1||end>=size)hv_fail("slice 区间越界");if(size&&start>end)length=1+(start-end-1)/(-step);}
    HValue result=tensor_view(source);HObject *view=result.as.obj;view->as.tensor.shape[axis]=length;view->as.tensor.count=shape_count(view->as.tensor.rank,view->as.tensor.shape);if(length)view->as.tensor.offset=add_i64(source->as.tensor.offset,mul_i64(start,source->as.tensor.strides[axis]));view->as.tensor.strides[axis]=mul_i64(source->as.tensor.strides[axis],step);return result;
}
static int64_t coordinate_axis(HObject *tensor,int64_t index,int64_t axis){for(int64_t dimension=tensor->as.tensor.rank;dimension-->axis+1;)index/=tensor->as.tensor.shape[dimension];return index%tensor->as.tensor.shape[axis];}
static HValue tensor_concat(HObject *left,HObject *right,int64_t axis){axis=axis_value(left,axis);if(left->as.tensor.rank!=right->as.tensor.rank)hv_fail("concat rank 不匹配");int64_t *shape=allocate(checked_size((size_t)left->as.tensor.rank,sizeof(int64_t)));for(int64_t dimension=0;dimension<left->as.tensor.rank;dimension++){shape[dimension]=left->as.tensor.shape[dimension];if(dimension!=axis&&shape[dimension]!=right->as.tensor.shape[dimension])hv_fail("concat shape 不匹配");}shape[axis]=add_i64(shape[axis],right->as.tensor.shape[axis]);HValue result=tensor_new(left->as.tensor.rank,shape,promoted(tensor_dtype(left),tensor_dtype(right)));release(shape);shape=result.as.obj->as.tensor.shape;if(!result.as.obj->as.tensor.count)return result;int64_t inner=1;for(int64_t dimension=left->as.tensor.rank;dimension-->axis+1;)inner=mul_i64(inner,shape[dimension]);for(int64_t index=0;index<result.as.obj->as.tensor.count;index++){int64_t coordinate=coordinate_axis(result.as.obj,index,axis),outer=index/(shape[axis]*inner),tail=index%inner;HObject *source=coordinate<left->as.tensor.shape[axis]?left:right;int64_t own=coordinate<left->as.tensor.shape[axis]?coordinate:coordinate-left->as.tensor.shape[axis];copy_element(result.as.obj,index,source,(outer*source->as.tensor.shape[axis]+own)*inner+tail);}return result;}
static HValue tensor_gather(HObject *source,HValue indices,int64_t axis){
    axis=axis_value(source,axis);
    require(indices,H_LIST);
    int64_t rank=source->as.tensor.rank;
    int64_t *shape=allocate(checked_size((size_t)rank,sizeof(int64_t)));
    memcpy(shape,source->as.tensor.shape,(size_t)rank*sizeof(int64_t));
    int64_t selected=hv_length(indices);
    shape[axis]=selected;
    for(int64_t item=0;item<selected;item++){
        int64_t coordinate=list_int(indices,item);
        if(coordinate<0||coordinate>=source->as.tensor.shape[axis])hv_fail("gather 索引越界");
    }
    int dtype=tensor_dtype(source);
    HValue result=tensor_allocate(rank,shape,dtype,0);
    release(shape);
    if(!result.as.obj->as.tensor.count)return result;
    int64_t inner=1;
    for(int64_t dimension=axis+1;dimension<rank;dimension++)inner=mul_i64(inner,source->as.tensor.shape[dimension]);
    if(axis==0&&tensor_contiguous(source)){
        size_t width=dtype_size(dtype),row_bytes=(size_t)inner*width;
        const char *input=(const char*)source->as.tensor.storage->as.storage.data+(size_t)source->as.tensor.offset*width;
        char *output=result.as.obj->as.tensor.storage->as.storage.data;
        for(int64_t row=0;row<selected;row++)
            memcpy(output+(size_t)row*row_bytes,input+(size_t)list_int(indices,row)*row_bytes,row_bytes);
        return result;
    }
    for(int64_t index=0;index<result.as.obj->as.tensor.count;index++){
        int64_t coordinate=index/inner%selected,outer=index/(selected*inner),tail=index%inner;
        int64_t from=list_int(indices,coordinate);
        copy_element(result.as.obj,index,source,(outer*source->as.tensor.shape[axis]+from)*inner+tail);
    }
    return result;
}
static HValue tensor_scatter(HObject *destination,HValue indices,HObject *source,int64_t axis){axis=axis_value(destination,axis);require(indices,H_LIST);if(destination->as.tensor.rank!=source->as.tensor.rank)hv_fail("scatter_add rank 不匹配");for(int64_t dimension=0;dimension<source->as.tensor.rank;dimension++)if(source->as.tensor.shape[dimension]!=(dimension==axis?hv_length(indices):destination->as.tensor.shape[dimension]))hv_fail("scatter_add shape 不匹配");for(int64_t item=0;item<hv_length(indices);item++){int64_t coordinate=list_int(indices,item);if(coordinate<0||coordinate>=destination->as.tensor.shape[axis])hv_fail("scatter_add 索引越界");}HValue result=tensor_clone(destination);if(!source->as.tensor.count)return result;int64_t inner=1;for(int64_t dimension=source->as.tensor.rank;dimension-->axis+1;)inner=mul_i64(inner,source->as.tensor.shape[dimension]);for(int64_t index=0;index<source->as.tensor.count;index++){int64_t coordinate=coordinate_axis(source,index,axis),outer=index/(source->as.tensor.shape[axis]*inner),tail=index%inner,to=(outer*destination->as.tensor.shape[axis]+list_int(indices,coordinate))*inner+tail;if(tensor_dtype(destination)==D_I64&&tensor_dtype(source)==D_I64){int64_t *data=result.as.obj->as.tensor.storage->as.storage.data;data[to]=add_i64(data[to],((int64_t*)source->as.tensor.storage->as.storage.data)[tensor_offset(source,index)]);}else tensor_write(result.as.obj,to,tensor_read(result.as.obj,to)+tensor_read(source,index));}return result;}
static HValue tensor_broadcast(HObject *source,HValue target){require(target,H_LIST);int64_t rank=hv_length(target);if(rank<source->as.tensor.rank)hv_fail("broadcast_to rank 不匹配");HValue result=tensor_view(source);HObject *view=result.as.obj;release(view->as.tensor.shape);release(view->as.tensor.strides);view->as.tensor.shape=allocate(checked_size((size_t)rank,sizeof(int64_t)));view->as.tensor.strides=allocate(checked_size((size_t)rank,sizeof(int64_t)));view->as.tensor.rank=rank;for(int64_t axis=0;axis<rank;axis++){int64_t own=axis-(rank-source->as.tensor.rank),size=own<0?1:source->as.tensor.shape[own],desired=list_int(target,axis);if(desired<0||(size!=1&&size!=desired))hv_fail("broadcast_to shape 不兼容");view->as.tensor.shape[axis]=desired;view->as.tensor.strides[axis]=own<0||size==1?0:source->as.tensor.strides[own];}view->as.tensor.count=shape_count(rank,view->as.tensor.shape);return result;}
static int tensor_reduce_shape_f64(HObject *source,HObject *output){
    if(tensor_dtype(source)!=D_F64||!tensor_contiguous(source))return 0;
    int64_t count=source->as.tensor.count;
    if(!count)return 1;
    const double *input=(const double*)source->as.tensor.storage->as.storage.data+source->as.tensor.offset;
    double *result=output->as.tensor.storage->as.storage.data;
    if(count==output->as.tensor.count){
        for(int64_t index=0;index<count;index++)result[index]=0.0+input[index];
        return 1;
    }
    int64_t rank=output->as.tensor.rank;
    if(source->as.tensor.rank!=2||rank==0||(rank==2&&output->as.tensor.shape[0]!=1))return 0;
    int64_t columns=source->as.tensor.shape[1];
    if(output->as.tensor.shape[rank-1]!=columns)return 0;
    for(int64_t row=0;row<source->as.tensor.shape[0];row++)
        for(int64_t column=0;column<columns;column++)result[column]+=input[row*columns+column];
    return 1;
}
static HValue tensor_reduce_shape(HObject *source,HValue target){HValue result=tensor_from_shape(target,tensor_dtype(source)==D_BOOL?D_I64:tensor_dtype(source));HObject *output=result.as.obj;int64_t rank=source->as.tensor.rank;if(output->as.tensor.rank>rank)hv_fail("reduce_to_shape rank 不匹配");for(int64_t axis=0;axis<rank;axis++){int64_t own=axis-(rank-output->as.tensor.rank),size=own<0?1:output->as.tensor.shape[own];if(size!=1&&size!=source->as.tensor.shape[axis])hv_fail("reduce_to_shape shape 不兼容");}if(tensor_reduce_shape_f64(source,output))return result;for(int64_t index=0;index<source->as.tensor.count;index++){int64_t remainder=index,to=0,stride=1;for(int64_t axis=rank;axis-->0;){int64_t coordinate=remainder%source->as.tensor.shape[axis];remainder/=source->as.tensor.shape[axis];int64_t own=axis-(rank-output->as.tensor.rank);if(own>=0){if(output->as.tensor.shape[own]!=1)to+=coordinate*stride;stride=mul_i64(stride,output->as.tensor.shape[own]?output->as.tensor.shape[own]:1);}}if(tensor_dtype(output)==D_I64){int64_t *data=output->as.tensor.storage->as.storage.data,exact=tensor_dtype(source)==D_I64?((int64_t*)source->as.tensor.storage->as.storage.data)[tensor_offset(source,index)]:(int64_t)tensor_read(source,index);data[to]=add_i64(data[to],exact);}else tensor_write(output,to,tensor_read(output,to)+tensor_read(source,index));}return result;}
#ifdef __clang__
#pragma STDC FP_CONTRACT OFF
#endif
typedef enum MatmulDenseMode {
    MATMUL_DENSE_OFF, MATMUL_DENSE_COMPENSATED, MATMUL_DENSE_FMA, MATMUL_DENSE_FMA_F32, MATMUL_DENSE_SPARSE_LEFT
} MatmulDenseMode;
typedef struct MatmulJob {
    HObject *left,*right,*output;
    int64_t begin,end,m,n,k,batch_rank;
    int64_t *batch_shape;
    int blocked;
    MatmulDenseMode dense_mode;
    double *packed_left;
} MatmulJob;
#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
enum { MATMUL_PANEL_COLUMNS=8, MATMUL_PANEL_MAX_INNER=512 };
static int matmul_has_fma(void){
    static int supported=-1;
    if(supported<0){
        unsigned int a,b,c,d;
        unsigned int required=bit_AVX|bit_OSXSAVE|bit_FMA;
        supported=0;
        if(__get_cpuid(1,&a,&b,&c,&d)&&(c&required)==required){
            __asm__ volatile("xgetbv":"=a"(a),"=d"(d):"c"(0));
            supported=(a&6)==6;
        }
    }
    return supported;
}

__attribute__((target("avx")))
static int matmul_max_exponent(const double *values,int64_t count,int *exponent){
    __m256d magnitude=_mm256_setzero_pd(),invalid=_mm256_setzero_pd();
    __m256d sign=_mm256_set1_pd(-0.0),limit=_mm256_set1_pd(DBL_MAX);
    int64_t index=0;
    for(;count-index>=4;index+=4){
        __m256d absolute=_mm256_andnot_pd(sign,_mm256_loadu_pd(values+index));
        invalid=_mm256_or_pd(invalid,_mm256_cmp_pd(absolute,limit,_CMP_NLE_UQ));
        magnitude=_mm256_max_pd(magnitude,absolute);
    }
    if(_mm256_movemask_pd(invalid))return 0;
    double lanes[4];
    _mm256_storeu_pd(lanes,magnitude);
    double maximum=fmax(fmax(lanes[0],lanes[1]),fmax(lanes[2],lanes[3]));
    for(;index<count;index++){
        if(!isfinite(values[index]))return 0;
        maximum=fmax(maximum,fabs(values[index]));
    }
    frexp(maximum,exponent);
    return 1;
}

__attribute__((target("avx")))
static int matmul_on_binary_grid(const double *values,int64_t count,int power){
    double scale=ldexp(1.0,power),unscale=ldexp(1.0,-power);
    __m256d factor=_mm256_set1_pd(scale),inverse=_mm256_set1_pd(unscale);
    int64_t index=0;
    for(;count-index>=4;index+=4){
        __m256d input=_mm256_loadu_pd(values+index);
        __m256d scaled=_mm256_mul_pd(input,factor);
        __m256d integral=_mm256_round_pd(scaled,_MM_FROUND_TO_ZERO|_MM_FROUND_NO_EXC);
        if(_mm256_movemask_pd(_mm256_cmp_pd(input,_mm256_mul_pd(integral,inverse),_CMP_NEQ_UQ)))return 0;
    }
    for(;index<count;index++){
        double scaled=values[index]*scale;
        if(values[index]!=trunc(scaled)*unscale)return 0;
    }
    return 1;
}

static int matmul_is_dense_matrix(HObject *tensor){
    const int64_t *stride=tensor->as.tensor.strides,*shape=tensor->as.tensor.shape;
    return (stride[1]==1&&stride[0]==shape[1])||(stride[0]==1&&stride[1]==shape[0]);
}

__attribute__((target("avx")))
static int matmul_f32_exponent(HObject *tensor,int *exponent){
    const float *values=(const float*)tensor->as.tensor.storage->as.storage.data+tensor->as.tensor.offset;
    int64_t count=tensor->as.tensor.count,index=0;
    __m256 magnitude=_mm256_setzero_ps(),invalid=magnitude;
    __m256 sign=_mm256_set1_ps(-0.0f),limit=_mm256_set1_ps(FLT_MAX);
    for(;count-index>=8;index+=8){
        __m256 absolute=_mm256_andnot_ps(sign,_mm256_loadu_ps(values+index));
        invalid=_mm256_or_ps(invalid,_mm256_cmp_ps(absolute,limit,_CMP_NLE_UQ));
        magnitude=_mm256_max_ps(magnitude,absolute);
    }
    if(_mm256_movemask_ps(invalid))return 0;
    float lanes[8],maximum=0;
    _mm256_storeu_ps(lanes,magnitude);
    for(int lane=0;lane<8;lane++)if(lanes[lane]>maximum)maximum=lanes[lane];
    for(;index<count;index++){
        float value=fabsf(values[index]);
        if(!isfinite(value))return 0;
        if(value>maximum)maximum=value;
    }
    frexpf(maximum,exponent);
    return 1;
}

__attribute__((target("avx")))
static int matmul_sparse_left(const double *values,int64_t count){
    static const unsigned char counts[16]={0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};
    int64_t remaining=count/8,index=0;
    __m256d zero=_mm256_setzero_pd();
    if(count>=8){
        __m256d first=_mm256_cmp_pd(_mm256_loadu_pd(values),zero,_CMP_NEQ_OQ);
        __m256d second=_mm256_cmp_pd(_mm256_loadu_pd(values+4),zero,_CMP_NEQ_OQ);
        if(_mm256_movemask_pd(_mm256_and_pd(first,second))==15)return 0;
    }
    for(;count-index>=4;index+=4){
        int mask=_mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(values+index),zero,_CMP_NEQ_OQ));
        remaining-=counts[mask];
        if(remaining<0)return 0;
    }
    for(;index<count;index++)if(values[index]!=0&&--remaining<0)return 0;
    return 1;
}

static MatmulDenseMode matmul_dense_mode(MatmulJob *job){
    HObject *left=job->left,*right=job->right;
    if(!job->blocked||job->batch_rank||job->m<8||job->n<MATMUL_PANEL_COLUMNS||job->k<8||job->k>MATMUL_PANEL_MAX_INNER||
       !matmul_is_dense_matrix(left)||!matmul_is_dense_matrix(right)||!matmul_has_fma()||
       (_mm_getcsr()&(_MM_ROUND_MASK|_MM_FLUSH_ZERO_MASK|_MM_DENORMALS_ZERO_MASK)))return MATMUL_DENSE_OFF;
    int left_exponent,right_exponent;
    int growth=64-__builtin_clzll((unsigned long long)(job->k-1));
    int left_dtype=tensor_dtype(left),right_dtype=tensor_dtype(right);
    if(left_dtype==D_F32&&right_dtype==D_F32&&matmul_fast&&job->n>=16&&
       left->as.tensor.strides[1]==1&&right->as.tensor.strides[1]==1){
        if(matmul_f32_exponent(left,&left_exponent)&&matmul_f32_exponent(right,&right_exponent)&&
           left_exponent+right_exponent+growth<=FLT_MAX_EXP-1)return MATMUL_DENSE_FMA_F32;
        return MATMUL_DENSE_OFF;
    }
    if(left_dtype!=D_F64||right_dtype!=D_F64)return MATMUL_DENSE_OFF;
    const double *first=(const double*)left->as.tensor.storage->as.storage.data+left->as.tensor.offset;
    const double *second=(const double*)right->as.tensor.storage->as.storage.data+right->as.tensor.offset;
    if(!matmul_max_exponent(first,left->as.tensor.count,&left_exponent)||
       !matmul_max_exponent(second,right->as.tensor.count,&right_exponent)||
       left_exponent+right_exponent+growth>DBL_MAX_EXP-1)return MATMUL_DENSE_OFF;
    if(matmul_fast){
        if(right->as.tensor.strides[1]==1&&matmul_sparse_left(first,left->as.tensor.count))return MATMUL_DENSE_SPARSE_LEFT;
        return MATMUL_DENSE_FMA;
    }
    int left_bits=(DBL_MANT_DIG-growth)/2,right_bits=DBL_MANT_DIG-growth-left_bits;
    if(left_exponent < -450||left_exponent > 450||right_exponent < -450||right_exponent > 450||
       !matmul_on_binary_grid(first,left->as.tensor.count,left_bits-left_exponent)||
       !matmul_on_binary_grid(second,right->as.tensor.count,right_bits-right_exponent))return MATMUL_DENSE_COMPENSATED;
    return MATMUL_DENSE_FMA;
}

#define MATMUL_FMA_ROW(s0,s1,source,vector,broadcast,fma) do { \
    vector a=broadcast(source); \
    s0=fma(a,b0,s0);s1=fma(a,b1,s1); \
} while(0)
#define MATMUL_PANEL_STORE(s0,s1,destination,store,lanes) do { \
    store(destination,s0);store((destination)+lanes,s1); \
} while(0)
#define DEFINE_MATMUL_FMA_PANEL(name,number,vector,lanes,zero,load,broadcast,fma,store) \
__attribute__((target("avx,fma"))) \
static void name(const number *left,const number *panel,number *output,int64_t rows,int64_t inner,int64_t columns){ \
    int64_t row=0; \
    for(;rows-row>=6;row+=6){ \
        vector sum00=zero(),sum01=sum00,sum10=sum00,sum11=sum00; \
        vector sum20=sum00,sum21=sum00,sum30=sum00,sum31=sum00; \
        vector sum40=sum00,sum41=sum00,sum50=sum00,sum51=sum00; \
        const number *first=left+row*inner; \
        for(int64_t index=0;index<inner;index++){ \
            const number *panel_row=panel+index*(2*lanes); \
            vector b0=load(panel_row),b1=load(panel_row+lanes); \
            MATMUL_FMA_ROW(sum00,sum01,first+index,vector,broadcast,fma); \
            MATMUL_FMA_ROW(sum10,sum11,first+inner+index,vector,broadcast,fma); \
            MATMUL_FMA_ROW(sum20,sum21,first+2*inner+index,vector,broadcast,fma); \
            MATMUL_FMA_ROW(sum30,sum31,first+3*inner+index,vector,broadcast,fma); \
            MATMUL_FMA_ROW(sum40,sum41,first+4*inner+index,vector,broadcast,fma); \
            MATMUL_FMA_ROW(sum50,sum51,first+5*inner+index,vector,broadcast,fma); \
        } \
        number *target=output+row*columns; \
        MATMUL_PANEL_STORE(sum00,sum01,target,store,lanes); \
        MATMUL_PANEL_STORE(sum10,sum11,target+columns,store,lanes); \
        MATMUL_PANEL_STORE(sum20,sum21,target+2*columns,store,lanes); \
        MATMUL_PANEL_STORE(sum30,sum31,target+3*columns,store,lanes); \
        MATMUL_PANEL_STORE(sum40,sum41,target+4*columns,store,lanes); \
        MATMUL_PANEL_STORE(sum50,sum51,target+5*columns,store,lanes); \
    } \
    for(;row<rows;row++){ \
        vector sum0=zero(),sum1=sum0; \
        for(int64_t index=0;index<inner;index++){ \
            const number *panel_row=panel+index*(2*lanes); \
            vector b0=load(panel_row),b1=load(panel_row+lanes); \
            MATMUL_FMA_ROW(sum0,sum1,left+row*inner+index,vector,broadcast,fma); \
        } \
        MATMUL_PANEL_STORE(sum0,sum1,output+row*columns,store,lanes); \
    } \
}
DEFINE_MATMUL_FMA_PANEL(matmul_fma_panel,double,__m256d,4,_mm256_setzero_pd,_mm256_load_pd,_mm256_broadcast_sd,_mm256_fmadd_pd,_mm256_storeu_pd)
DEFINE_MATMUL_FMA_PANEL(matmul_fma_panel_f32,float,__m256,8,_mm256_setzero_ps,_mm256_load_ps,_mm256_broadcast_ss,_mm256_fmadd_ps,_mm256_storeu_ps)
#undef DEFINE_MATMUL_FMA_PANEL
#undef MATMUL_PANEL_STORE
#undef MATMUL_FMA_ROW

#define MATMUL_STORE_ROW(s0,s1,destination) do { \
    _mm256_storeu_pd(destination,s0);_mm256_storeu_pd((destination)+4,s1); \
} while(0)
#define MATMUL_LOAD_PANEL(panel_row) \
    __m256d b0=_mm256_load_pd(panel_row),b1=_mm256_load_pd((panel_row)+4)

#define MATMUL_COMPENSATE(total,correction,value) do { \
    __m256d adjusted=_mm256_sub_pd(value,correction); \
    __m256d next=_mm256_add_pd(total,adjusted); \
    correction=_mm256_sub_pd(_mm256_sub_pd(next,total),adjusted); \
    total=next; \
} while(0)
__attribute__((target("avx")))
static void matmul_compensated_panel(const double *left,const double *panel,double *output,int64_t rows,int64_t inner,int64_t columns){
    int64_t row=0;
    for(;rows-row>=3;row+=3){
        __m256d sum00=_mm256_setzero_pd(),sum01=sum00,sum10=sum00,sum11=sum00,sum20=sum00,sum21=sum00;
        __m256d correction00=sum00,correction01=sum00,correction10=sum00,correction11=sum00,correction20=sum00,correction21=sum00;
        const double *first=left+row*inner;
        for(int64_t index=0;index<inner;index++){
            MATMUL_LOAD_PANEL(panel+index*MATMUL_PANEL_COLUMNS);
            __m256d a=_mm256_broadcast_sd(first+index);
            MATMUL_COMPENSATE(sum00,correction00,_mm256_mul_pd(a,b0));
            MATMUL_COMPENSATE(sum01,correction01,_mm256_mul_pd(a,b1));
            a=_mm256_broadcast_sd(first+inner+index);
            MATMUL_COMPENSATE(sum10,correction10,_mm256_mul_pd(a,b0));
            MATMUL_COMPENSATE(sum11,correction11,_mm256_mul_pd(a,b1));
            a=_mm256_broadcast_sd(first+2*inner+index);
            MATMUL_COMPENSATE(sum20,correction20,_mm256_mul_pd(a,b0));
            MATMUL_COMPENSATE(sum21,correction21,_mm256_mul_pd(a,b1));
        }
        MATMUL_STORE_ROW(sum00,sum01,output+row*columns);
        MATMUL_STORE_ROW(sum10,sum11,output+(row+1)*columns);
        MATMUL_STORE_ROW(sum20,sum21,output+(row+2)*columns);
    }
    for(;row<rows;row++){
        __m256d sum0=_mm256_setzero_pd(),sum1=sum0,correction0=sum0,correction1=sum0;
        for(int64_t index=0;index<inner;index++){
            MATMUL_LOAD_PANEL(panel+index*MATMUL_PANEL_COLUMNS);
            __m256d a=_mm256_broadcast_sd(left+row*inner+index);
            MATMUL_COMPENSATE(sum0,correction0,_mm256_mul_pd(a,b0));
            MATMUL_COMPENSATE(sum1,correction1,_mm256_mul_pd(a,b1));
        }
        MATMUL_STORE_ROW(sum0,sum1,output+row*columns);
    }
}
#undef MATMUL_COMPENSATE
#undef MATMUL_LOAD_PANEL
#undef MATMUL_STORE_ROW

__attribute__((target("avx")))
static void matmul_pack_transpose(const double *source,double *target,int64_t rows,int64_t columns,
                                  int64_t source_stride,int64_t target_stride){
    int64_t row=0;
    for(;rows-row>=4;row+=4){
        int64_t column=0;
        for(;columns-column>=4;column+=4){
            const double *input=source+row*source_stride+column;
            __m256d a=_mm256_loadu_pd(input),b=_mm256_loadu_pd(input+source_stride);
            __m256d c=_mm256_loadu_pd(input+2*source_stride),d=_mm256_loadu_pd(input+3*source_stride);
            __m256d low_ab=_mm256_unpacklo_pd(a,b),high_ab=_mm256_unpackhi_pd(a,b);
            __m256d low_cd=_mm256_unpacklo_pd(c,d),high_cd=_mm256_unpackhi_pd(c,d);
            double *output=target+column*target_stride+row;
            _mm256_storeu_pd(output,_mm256_permute2f128_pd(low_ab,low_cd,0x20));
            _mm256_storeu_pd(output+target_stride,_mm256_permute2f128_pd(high_ab,high_cd,0x20));
            _mm256_storeu_pd(output+2*target_stride,_mm256_permute2f128_pd(low_ab,low_cd,0x31));
            _mm256_storeu_pd(output+3*target_stride,_mm256_permute2f128_pd(high_ab,high_cd,0x31));
        }
        for(;column<columns;column++)
            for(int64_t lane=0;lane<4;lane++)target[column*target_stride+row+lane]=source[(row+lane)*source_stride+column];
    }
    for(;row<rows;row++)
        for(int64_t column=0;column<columns;column++)target[column*target_stride+row]=source[row*source_stride+column];
}

static void matmul_pack_left(MatmulJob *job){
    HObject *tensor=job->left;
    if(tensor->as.tensor.strides[1]==1)return;
    const double *source=(const double*)tensor->as.tensor.storage->as.storage.data+tensor->as.tensor.offset;
    job->packed_left=allocate_storage(checked_size((size_t)tensor->as.tensor.count,sizeof(double)));
    for(int64_t start=0;start<job->m;){
        int64_t rows=job->m-start>16?16:job->m-start;
        matmul_pack_transpose(source+start,job->packed_left+start*job->k,job->k,rows,job->m,job->k);
        start+=rows;
    }
}

__attribute__((target("avx,fma")))
static void matmul_sparse_left_f64(MatmulJob *job){
    int64_t inner=job->k,columns=job->n;
    const double *left=job->packed_left;
    if(!left)left=(const double*)job->left->as.tensor.storage->as.storage.data+job->left->as.tensor.offset;
    const double *right=(const double*)job->right->as.tensor.storage->as.storage.data+job->right->as.tensor.offset;
    double *output=job->output->as.tensor.storage->as.storage.data;
    for(int64_t row=job->begin;row<job->end;row++){
        double *target=output+row*columns;
        memset(target,0,(size_t)columns*sizeof(double));
        for(int64_t index=0;index<inner;index++){
            double coefficient=left[row*inner+index];
            if(coefficient==0)continue;
            const double *source=right+index*columns;
            __m256d weight=_mm256_set1_pd(coefficient);
            int64_t column=0;
            for(;columns-column>=4;column+=4){
                __m256d values=_mm256_loadu_pd(source+column),total=_mm256_loadu_pd(target+column);
                _mm256_storeu_pd(target+column,_mm256_fmadd_pd(weight,values,total));
            }
            for(;column<columns;column++)target[column]=fma(coefficient,source[column],target[column]);
        }
    }
}

static void matmul_dense_f64(MatmulJob *job){
    int64_t inner=job->k,columns=job->n,rows=job->end-job->begin;
    const double *left=job->packed_left;
    if(!left)left=(const double*)job->left->as.tensor.storage->as.storage.data+job->left->as.tensor.offset;
    left+=job->begin*inner;
    const double *right=(const double*)job->right->as.tensor.storage->as.storage.data+job->right->as.tensor.offset;
    int64_t right_row=job->right->as.tensor.strides[0],right_column=job->right->as.tensor.strides[1];
    double *output=(double*)job->output->as.tensor.storage->as.storage.data+job->begin*columns;
    _Alignas(64) double panel[MATMUL_PANEL_MAX_INNER*MATMUL_PANEL_COLUMNS];
    int64_t column=0;
    for(;columns-column>=MATMUL_PANEL_COLUMNS;column+=MATMUL_PANEL_COLUMNS){
        if(right_column==1){
            for(int64_t index=0;index<inner;index++)
                memcpy(panel+index*MATMUL_PANEL_COLUMNS,right+index*right_row+column,MATMUL_PANEL_COLUMNS*sizeof(double));
        }else{
            matmul_pack_transpose(right+column*right_column,panel,MATMUL_PANEL_COLUMNS,inner,right_column,MATMUL_PANEL_COLUMNS);
        }
        if(job->dense_mode==MATMUL_DENSE_FMA)matmul_fma_panel(left,panel,output+column,rows,inner,columns);
        else matmul_compensated_panel(left,panel,output+column,rows,inner,columns);
    }
    for(int64_t row=0;row<rows;row++)for(int64_t tail=column;tail<columns;tail++){
        double total=0,correction=0;
        for(int64_t index=0;index<inner;index++){
            double adjusted=left[row*inner+index]*right[index*right_row+tail*right_column]-correction;
            double next=total+adjusted;
            correction=(next-total)-adjusted;
            total=next;
        }
        output[row*columns+tail]=total;
    }
}
static void matmul_dense_f32(MatmulJob *job){
    int64_t inner=job->k,columns=job->n,rows=job->end-job->begin;
    const float *left=(const float*)job->left->as.tensor.storage->as.storage.data+job->left->as.tensor.offset+job->begin*inner;
    const float *right=(const float*)job->right->as.tensor.storage->as.storage.data+job->right->as.tensor.offset;
    float *output=(float*)job->output->as.tensor.storage->as.storage.data+job->begin*columns;
    _Alignas(64) float panel[MATMUL_PANEL_MAX_INNER*16];
    int64_t column=0;
    for(;columns-column>=16;column+=16){
        for(int64_t index=0;index<inner;index++)memcpy(panel+index*16,right+index*columns+column,16*sizeof(float));
        matmul_fma_panel_f32(left,panel,output+column,rows,inner,columns);
    }
    for(int64_t row=0;row<rows;row++)for(int64_t tail=column;tail<columns;tail++){
        double total=0,correction=0;
        for(int64_t index=0;index<inner;index++){
            double adjusted=(double)left[row*inner+index]*right[index*columns+tail]-correction;
            double next=total+adjusted;
            correction=(next-total)-adjusted;
            total=next;
        }
        output[row*columns+tail]=(float)total;
    }
}
#endif
#define MATMUL_FLOAT_COLUMNS(right_index) do { \
    for(int64_t column=0;column<width;column++){ \
        double b=second[right_index],value=a*b,adjusted=value-corrections[column],next=totals[column]+adjusted; \
        corrections[column]=isfinite(next)?(next-totals[column])-adjusted:0;totals[column]=next; \
    } \
} while(0)
#define MATMUL_FLOAT_TYPES(left_type,right_type,output_type) do { \
    const left_type *first=(const left_type*)left->as.tensor.storage->as.storage.data; \
    const right_type *second=(const right_type*)right->as.tensor.storage->as.storage.data; \
    output_type *target=job->output->as.tensor.storage->as.storage.data; \
    for(int64_t block=0;block<job->n;){ \
        int64_t width=job->n-block>32?32:job->n-block; \
        double totals[32]={0},corrections[32]={0}; \
        for(int64_t inner=0;inner<job->k;inner++){ \
            double a=first[lo+inner*ls]; \
            int64_t origin=ro+inner*rs+block*cs; \
            if(cs==1)MATMUL_FLOAT_COLUMNS(origin+column);else MATMUL_FLOAT_COLUMNS(origin+column*cs); \
        } \
        for(int64_t column=0;column<width;column++)target[row*job->n+block+column]=totals[column]; \
        block+=width; \
    } \
} while(0)
static int matmul_float_row(MatmulJob *job,int64_t row,int64_t lo,int64_t ro){
    HObject *left=job->left,*right=job->right;
    int ld=tensor_dtype(left),rd=tensor_dtype(right);
    if(!job->blocked||(ld!=D_F64&&ld!=D_F32)||(rd!=D_F64&&rd!=D_F32))return 0;
    int64_t ls=left->as.tensor.strides[left->as.tensor.rank-1],rs=right->as.tensor.strides[right->as.tensor.rank-2],cs=right->as.tensor.strides[right->as.tensor.rank-1];
    if(ld==D_F64){if(rd==D_F64)MATMUL_FLOAT_TYPES(double,double,double);else MATMUL_FLOAT_TYPES(double,float,double);}
    else {if(rd==D_F64)MATMUL_FLOAT_TYPES(float,double,double);else MATMUL_FLOAT_TYPES(float,float,float);}
    return 1;
}
#undef MATMUL_FLOAT_TYPES
#undef MATMUL_FLOAT_COLUMNS
static void matmul_run(MatmulJob *job){HObject *left=job->left,*right=job->right,*output=job->output;int64_t lr=left->as.tensor.rank,rr=right->as.tensor.rank;for(int64_t row=job->begin;row<job->end;row++){int64_t batch=row/job->m,m=row%job->m,remainder=batch,lo=left->as.tensor.offset,ro=right->as.tensor.offset;for(int64_t axis=job->batch_rank;axis-->0;){int64_t coordinate=remainder%job->batch_shape[axis];remainder/=job->batch_shape[axis];int64_t la=axis-(job->batch_rank-(lr-2)),ra=axis-(job->batch_rank-(rr-2));if(la>=0&&left->as.tensor.shape[la]!=1)lo+=coordinate*left->as.tensor.strides[la];if(ra>=0&&right->as.tensor.shape[ra]!=1)ro+=coordinate*right->as.tensor.strides[ra];}lo+=m*left->as.tensor.strides[lr-2];if(matmul_float_row(job,row,lo,ro))continue;for(int64_t block=0;block<job->n;block+=job->blocked?32:job->n){int64_t end=job->blocked&&job->n-block>32?block+32:job->n;double totals[32],corrections[32];if(job->blocked){memset(totals,0,sizeof(totals));memset(corrections,0,sizeof(corrections));}for(int64_t column=block;column<end;column++){int64_t target=row*job->n+column;if(tensor_dtype(output)==D_I64){int64_t total=0;for(int64_t k=0;k<job->k;k++){int64_t a=tensor_dtype(left)==D_I64?((int64_t*)left->as.tensor.storage->as.storage.data)[lo+k*left->as.tensor.strides[lr-1]]:(int64_t)read_offset(left,lo+k*left->as.tensor.strides[lr-1]);int64_t b=tensor_dtype(right)==D_I64?((int64_t*)right->as.tensor.storage->as.storage.data)[ro+k*right->as.tensor.strides[rr-2]+column*right->as.tensor.strides[rr-1]]:(int64_t)read_offset(right,ro+k*right->as.tensor.strides[rr-2]+column*right->as.tensor.strides[rr-1]);total=add_i64(total,mul_i64(a,b));}((int64_t*)output->as.tensor.storage->as.storage.data)[target]=total;}else if(!job->blocked){double total=0,correction=0;for(int64_t k=0;k<job->k;k++){double value=read_offset(left,lo+k*left->as.tensor.strides[lr-1])*read_offset(right,ro+k*right->as.tensor.strides[rr-2]+column*right->as.tensor.strides[rr-1]);double adjusted=value-correction,next=total+adjusted;correction=isfinite(next)?(next-total)-adjusted:0;total=next;}tensor_write(output,target,total);}}
            if(job->blocked&&tensor_dtype(output)!=D_I64){for(int64_t kblock=0;kblock<job->k;kblock+=32){int64_t kend=job->k-kblock>32?kblock+32:job->k;for(int64_t k=kblock;k<kend;k++){double a=read_offset(left,lo+k*left->as.tensor.strides[lr-1]);for(int64_t column=block;column<end;column++){int64_t local=column-block;double value=a*read_offset(right,ro+k*right->as.tensor.strides[rr-2]+column*right->as.tensor.strides[rr-1]),adjusted=value-corrections[local],next=totals[local]+adjusted;corrections[local]=isfinite(next)?(next-totals[local])-adjusted:0;totals[local]=next;}}}for(int64_t column=block;column<end;column++)tensor_write(output,row*job->n+column,totals[column-block]);}
        }} }
static void matmul_rows(MatmulJob *job){
#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
    if(job->dense_mode==MATMUL_DENSE_FMA_F32){matmul_dense_f32(job);return;}
    if(job->dense_mode==MATMUL_DENSE_SPARSE_LEFT){matmul_sparse_left_f64(job);return;}
    if(job->dense_mode){matmul_dense_f64(job);return;}
#endif
    matmul_run(job);
}
#ifdef _WIN32
static DWORD WINAPI matmul_thread(LPVOID argument){matmul_rows((MatmulJob*)argument);return 0;}
#endif
static HValue tensor_matmul(HObject *left,HObject *right){int64_t lr=left->as.tensor.rank,rr=right->as.tensor.rank;if(lr<2||rr<2)hv_fail("matmul 需要至少二维张量");int64_t m=left->as.tensor.shape[lr-2],k=left->as.tensor.shape[lr-1],n=right->as.tensor.shape[rr-1];if(k!=right->as.tensor.shape[rr-2])hv_fail("matmul 内维度不匹配");int64_t rank=lr>rr?lr:rr;int64_t *shape=allocate(checked_size((size_t)rank,sizeof(int64_t)));for(int64_t axis=0;axis<rank-2;axis++){int64_t la=axis-(rank-lr),ra=axis-(rank-rr),a=la<0?1:left->as.tensor.shape[la],b=ra<0?1:right->as.tensor.shape[ra];if(a!=b&&a!=1&&b!=1)hv_fail("matmul 批维度不兼容");shape[axis]=a==1?b:a;}shape[rank-2]=m;shape[rank-1]=n;int dtype=promoted(tensor_dtype(left),tensor_dtype(right));if(dtype==D_BOOL)dtype=D_I64;HValue result=tensor_allocate(rank,shape,dtype,0);release(shape);if(!result.as.obj->as.tensor.count)return result;int64_t rows=result.as.obj->as.tensor.count/n;MatmulJob job={0};job.left=left;job.right=right;job.output=result.as.obj;job.end=rows;job.m=m;job.n=n;job.k=k;job.batch_rank=rank-2;job.blocked=optimized;job.batch_shape=result.as.obj->as.tensor.shape;
#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
    job.dense_mode=matmul_dense_mode(&job);
    if(job.dense_mode)matmul_pack_left(&job);
#endif
#ifdef _WIN32
    int workers=thread_count;
    if(workers>rows)workers=(int)rows;
    if(workers>1){
        HANDLE *handles=allocate((size_t)workers*sizeof(HANDLE));
        MatmulJob *jobs=allocate((size_t)workers*sizeof(MatmulJob));
        for(int index=0;index<workers;index++){
            jobs[index]=job;
            jobs[index].begin=(rows/workers)*index+(rows%workers)*index/workers;
            jobs[index].end=(rows/workers)*(index+1)+(rows%workers)*(index+1)/workers;
            handles[index]=CreateThread(NULL,0,matmul_thread,&jobs[index],0,NULL);
            if(!handles[index])hv_fail("创建计算线程失败");
        }
        for(int index=0;index<workers;index++){
            if(WaitForSingleObject(handles[index],INFINITE)!=WAIT_OBJECT_0)hv_fail("等待计算线程失败");
            CloseHandle(handles[index]);
        }
        release(handles);
        release(jobs);
    }else
#endif
    matmul_rows(&job);
    release_storage(job.packed_left);
    return result;
}
#ifdef __clang__
#pragma STDC FP_CONTRACT DEFAULT
#endif
static HValue tensor_softmax(HObject *source,int64_t axis){axis=axis_value(source,axis);int dtype=tensor_dtype(source)==D_F32?D_F32:D_F64;HValue result=tensor_new(source->as.tensor.rank,source->as.tensor.shape,dtype);if(!source->as.tensor.count)return result;int64_t width=source->as.tensor.shape[axis],inner=1,outer=1;for(int64_t dimension=0;dimension<axis;dimension++)outer=mul_i64(outer,source->as.tensor.shape[dimension]);for(int64_t dimension=axis+1;dimension<source->as.tensor.rank;dimension++)inner=mul_i64(inner,source->as.tensor.shape[dimension]);if(!width)return result;for(int64_t group=0;group<outer;group++)for(int64_t tail=0;tail<inner;tail++){double maximum=-INFINITY,total=0;int64_t positive=0;for(int64_t item=0;item<width;item++){double value=tensor_read(source,(group*width+item)*inner+tail);if(isnan(value))hv_fail("softmax 输入含 NaN");if(value>maximum)maximum=value;if(value==INFINITY)positive++;}for(int64_t item=0;item<width;item++){int64_t index=(group*width+item)*inner+tail;double value=tensor_read(source,index);double weight=positive?(value==INFINITY?1.0:0.0):maximum==-INFINITY?1.0:exp(value-maximum);tensor_write(result.as.obj,index,weight);total+=weight;}for(int64_t item=0;item<width;item++){int64_t index=(group*width+item)*inner+tail;tensor_write(result.as.obj,index,tensor_read(result.as.obj,index)/total);}}return result;}
static void convolution_shape(const int64_t *shape,int64_t kh,int64_t kw,int64_t stride,int64_t padding,int64_t *oh,int64_t *ow){if(kh<=0||kw<=0||stride<=0||padding<0)hv_fail("卷积窗口参数无效");int64_t height=sub_i64(add_i64(shape[2],mul_i64(2,padding)),kh),width=sub_i64(add_i64(shape[3],mul_i64(2,padding)),kw);*oh=height<0?0:add_i64(height/stride,1);*ow=width<0?0:add_i64(width/stride,1);}
static HValue tensor_unfold(HObject *source,int64_t kh,int64_t kw,int64_t stride,int64_t padding){if(source->as.tensor.rank!=4)hv_fail("unfold2d 需要 NCHW 张量");int64_t oh,ow;convolution_shape(source->as.tensor.shape,kh,kw,stride,padding,&oh,&ow);int64_t shape[3]={source->as.tensor.shape[0],mul_i64(source->as.tensor.shape[1],mul_i64(kh,kw)),mul_i64(oh,ow)};HValue result=tensor_new(3,shape,tensor_dtype(source));int64_t channels=source->as.tensor.shape[1],height=source->as.tensor.shape[2],width=source->as.tensor.shape[3];for(int64_t index=0;index<result.as.obj->as.tensor.count;index++){int64_t location=index%shape[2],channel=(index/shape[2])%shape[1],batch=index/(shape[2]*shape[1]),c=channel/(kh*kw),y=(location/ow)*stride-padding+(channel/kw)%kh,x=(location%ow)*stride-padding+channel%kw;if(y>=0&&y<height&&x>=0&&x<width)copy_element(result.as.obj,index,source,((batch*channels+c)*height+y)*width+x);}return result;}
static HValue tensor_fold(HObject *source,HValue target,int64_t kh,int64_t kw,int64_t stride,int64_t padding){HValue result=tensor_from_shape(target,tensor_dtype(source));HObject *output=result.as.obj;if(output->as.tensor.rank!=4||source->as.tensor.rank!=3)hv_fail("fold2d shape 无效");int64_t oh,ow;convolution_shape(output->as.tensor.shape,kh,kw,stride,padding,&oh,&ow);int64_t channels=output->as.tensor.shape[1],height=output->as.tensor.shape[2],width=output->as.tensor.shape[3],channel_width=mul_i64(channels,mul_i64(kh,kw)),locations=mul_i64(oh,ow);if(source->as.tensor.shape[0]!=output->as.tensor.shape[0]||source->as.tensor.shape[1]!=channel_width||source->as.tensor.shape[2]!=locations)hv_fail("fold2d 输入列 shape 不匹配");for(int64_t index=0;index<source->as.tensor.count;index++){int64_t location=index%locations,channel=(index/locations)%channel_width,batch=index/(locations*channel_width),c=channel/(kh*kw),y=(location/ow)*stride-padding+(channel/kw)%kh,x=(location%ow)*stride-padding+channel%kw;if(y>=0&&y<height&&x>=0&&x<width){int64_t to=((batch*channels+c)*height+y)*width+x;if(tensor_dtype(source)==D_I64){int64_t *data=output->as.tensor.storage->as.storage.data;data[to]=add_i64(data[to],((int64_t*)source->as.tensor.storage->as.storage.data)[tensor_offset(source,index)]);}else tensor_write(output,to,tensor_read(output,to)+tensor_read(source,index));}}return result;}
typedef struct SparseEntry { int64_t row,col,order; double value; } SparseEntry;
static int sparse_compare(const void *first,const void *second){const SparseEntry *left=first,*right=second;if(left->row!=right->row)return left->row<right->row?-1:1;if(left->col!=right->col)return left->col<right->col?-1:1;return left->order<right->order?-1:left->order>right->order?1:0;}
static HValue sparse_build(int64_t rows,int64_t cols,int64_t count,SparseEntry *entries,int coalesce,int csr){
    if(rows<0||cols<0||count<0)hv_fail("稀疏 shape 无效");
    for(int64_t index=0;index<count;index++)entries[index].order=index;
    qsort(entries,(size_t)count,sizeof(SparseEntry),sparse_compare);
    int64_t unique=0;
    for(int64_t index=0;index<count;index++){
        if(entries[index].row<0||entries[index].row>=rows||entries[index].col<0||entries[index].col>=cols)hv_fail("稀疏索引越界");
        if(unique&&entries[index].row==entries[unique-1].row&&entries[index].col==entries[unique-1].col){
            if(!coalesce)hv_fail("CSR 稀疏索引重复");
            entries[unique-1].value+=entries[index].value;
        }else entries[unique++]=entries[index];
    }
    HValue result=object_value(H_SPARSE);
    HObject *sparse=result.as.obj;
    sparse->as.sparse.csr=csr;
    sparse->as.sparse.rows=rows;sparse->as.sparse.cols=cols;sparse->as.sparse.count=unique;
    size_t row_bytes=checked_size((size_t)(csr?add_i64(rows,1):unique),sizeof(int64_t));
    sparse->as.sparse.row=allocate(row_bytes);
    if(csr)memset(sparse->as.sparse.row,0,row_bytes);
    sparse->as.sparse.col=allocate(checked_size((size_t)unique,sizeof(int64_t)));
    sparse->as.sparse.values=allocate(checked_size((size_t)unique,sizeof(double)));
    for(int64_t index=0;index<unique;index++){
        if(csr)sparse->as.sparse.row[entries[index].row+1]++;
        else sparse->as.sparse.row[index]=entries[index].row;
        sparse->as.sparse.col[index]=entries[index].col;
        sparse->as.sparse.values[index]=entries[index].value;
    }
    if(csr)for(int64_t row=1;row<=rows;row++)sparse->as.sparse.row[row]+=sparse->as.sparse.row[row-1];
    return result;
}
static HValue sparse_construct(int csr,HValue first,HValue columns,HValue values,int64_t rows,int64_t cols){require(first,H_LIST);require(columns,H_LIST);require(values,H_LIST);int64_t count=hv_length(values);if(rows<0||cols<0||hv_length(columns)!=count)hv_fail("稀疏参数不匹配");SparseEntry *entries=allocate(checked_size((size_t)count,sizeof(SparseEntry)));if(csr){if(rows==INT64_MAX||hv_length(first)!=rows+1||list_int(first,0)!=0||list_int(first,rows)!=count)hv_fail("CSR offsets 无效");for(int64_t row=0;row<rows;row++){int64_t start=list_int(first,row),end=list_int(first,row+1);if(start<0||end<start||end>count)hv_fail("CSR offsets 非单调或越界");for(int64_t index=start;index<end;index++)entries[index].row=row;}}else{if(hv_length(first)!=count)hv_fail("COO 索引数量不匹配");for(int64_t index=0;index<count;index++)entries[index].row=list_int(first,index);}for(int64_t index=0;index<count;index++){entries[index].col=list_int(columns,index);entries[index].value=hv_number(hv_index(values,hv_i64(index)));}HValue result=sparse_build(rows,cols,count,entries,!csr,csr);release(entries);return result;}
static int64_t sparse_row(HObject *sparse,int64_t entry){
    if(!sparse->as.sparse.csr)return sparse->as.sparse.row[entry];
    int64_t low=0,high=sparse->as.sparse.rows;
    while(low<high){
        int64_t middle=low+(high-low)/2;
        if(sparse->as.sparse.row[middle+1]<=entry)low=middle+1;
        else high=middle;
    }
    return low;
}
static HValue sparse_dense(HObject *sparse){int64_t shape[2]={sparse->as.sparse.rows,sparse->as.sparse.cols};HValue result=tensor_new(2,shape,D_F64);for(int64_t index=0;index<sparse->as.sparse.count;index++)tensor_write(result.as.obj,sparse_row(sparse,index)*shape[1]+sparse->as.sparse.col[index],sparse->as.sparse.values[index]);return result;}
static HValue sparse_transpose(HObject *sparse){int64_t count=sparse->as.sparse.count;SparseEntry *entries=allocate(checked_size((size_t)count,sizeof(SparseEntry)));for(int64_t index=0;index<count;index++){entries[index].row=sparse->as.sparse.col[index];entries[index].col=sparse_row(sparse,index);entries[index].value=sparse->as.sparse.values[index];}HValue result=sparse_build(sparse->as.sparse.cols,sparse->as.sparse.rows,count,entries,0,sparse->as.sparse.csr);release(entries);return result;}
static HValue sparse_matmul(HObject *sparse,HObject *dense){if(dense->as.tensor.rank!=2||dense->as.tensor.shape[0]!=sparse->as.sparse.cols)hv_fail("sparse_matmul shape 不匹配");int64_t shape[2]={sparse->as.sparse.rows,dense->as.tensor.shape[1]};HValue result=tensor_new(2,shape,D_F64);for(int64_t entry=0;entry<sparse->as.sparse.count;entry++)for(int64_t column=0;column<shape[1];column++){int64_t index=sparse_row(sparse,entry)*shape[1]+column;tensor_write(result.as.obj,index,tensor_read(result.as.obj,index)+sparse->as.sparse.values[entry]*tensor_read(dense,sparse->as.sparse.col[entry]*shape[1]+column));}return result;}
typedef struct Buffer { unsigned char *data; size_t size,capacity,cursor; HObject *format_path[25]; } Buffer;
static void buffer_append(Buffer *buffer,const void *data,size_t size){if(size>SIZE_MAX-buffer->size)hv_fail("缓冲区尺寸溢出");size_t needed=buffer->size+size;if(needed>buffer->capacity){size_t capacity=buffer->capacity?buffer->capacity:128;while(capacity<needed){if(capacity>SIZE_MAX/2){capacity=needed;break;}capacity*=2;}buffer->data=resize(buffer->data,capacity);buffer->capacity=capacity;}if(size)memcpy(buffer->data+buffer->size,data,size);buffer->size=needed;}
static void buffer_text(Buffer *buffer,const char *text){buffer_append(buffer,text,strlen(text));}
static void format_value(Buffer *buffer,HValue value,int depth){char number[128];if(depth>24){buffer_text(buffer,"<depth-limit>");return;}if(value.tag>=H_STRING){for(int ancestor=0;ancestor<depth;ancestor++)if(buffer->format_path[ancestor]==value.as.obj){buffer_text(buffer,"<cycle>");return;}buffer->format_path[depth]=value.as.obj;}else buffer->format_path[depth]=NULL;switch(value.tag){case H_VOID:buffer_text(buffer,"void");break;case H_I64:snprintf(number,sizeof(number),"%lld",(long long)value.as.i);buffer_text(buffer,number);break;case H_F32:case H_F64:snprintf(number,sizeof(number),value.tag==H_F32?"%.9g":"%.17g",value.as.f);buffer_text(buffer,number);break;case H_BOOL:buffer_text(buffer,value.as.i?"true":"false");break;case H_STRING:buffer_text(buffer,text_of(value));break;case H_LIST:buffer_text(buffer,"[");for(size_t index=0;index<value.as.obj->as.list.count;index++){if(index)buffer_text(buffer,", ");format_value(buffer,value.as.obj->as.list.items[index],depth+1);}buffer_text(buffer,"]");break;case H_RECORD:buffer_text(buffer,value.as.obj->as.record.type);buffer_text(buffer,"{");for(size_t index=0;index<value.as.obj->as.record.count;index++){if(index)buffer_text(buffer,", ");buffer_text(buffer,value.as.obj->as.record.names[index]);buffer_text(buffer,": ");format_value(buffer,value.as.obj->as.record.values[index],depth+1);}buffer_text(buffer,"}");break;case H_TENSOR:buffer_text(buffer,"tensor([");for(int64_t index=0;index<value.as.obj->as.tensor.count;index++){if(index)buffer_text(buffer,", ");if(tensor_dtype(value.as.obj)==D_I64)format_value(buffer,hv_i64(((int64_t*)value.as.obj->as.tensor.storage->as.storage.data)[tensor_offset(value.as.obj,index)]),depth+1);else format_value(buffer,hv_f64(tensor_read(value.as.obj,index)),depth+1);}buffer_text(buffer,"], shape=[");for(int64_t axis=0;axis<value.as.obj->as.tensor.rank;axis++){if(axis)buffer_text(buffer,", ");format_value(buffer,hv_i64(value.as.obj->as.tensor.shape[axis]),depth+1);}buffer_text(buffer,"], dtype=");buffer_text(buffer,dtype_name(tensor_dtype(value.as.obj)));buffer_text(buffer,")");break;case H_SPARSE:snprintf(number,sizeof(number),"sparse(rows=%lld, cols=%lld, nnz=%lld)",(long long)value.as.obj->as.sparse.rows,(long long)value.as.obj->as.sparse.cols,(long long)value.as.obj->as.sparse.count);buffer_text(buffer,number);break;case H_RESULT:buffer_text(buffer,value.as.obj->as.result.ok?"ok(":"err(");format_value(buffer,value.as.obj->as.result.ok?value.as.obj->as.result.value:value.as.obj->as.result.error,depth+1);buffer_text(buffer,")");break;default:hv_fail("无法格式化的值");}}
static uint64_t checksum(const unsigned char *data,size_t size){uint64_t hash=UINT64_C(14695981039346656037);for(size_t index=0;index<size;index++){hash^=data[index];hash*=UINT64_C(1099511628211);}return hash;}
static void put_unsigned(Buffer *buffer,uint64_t value,size_t width){unsigned char bytes[8];for(size_t index=0;index<width;index++)bytes[index]=(unsigned char)(value>>(index*8));buffer_append(buffer,bytes,width);}
static uint64_t take_unsigned(Buffer *buffer,size_t width){if(width>buffer->size-buffer->cursor)hv_fail("状态文件被截断");uint64_t value=0;for(size_t index=0;index<width;index++)value|=(uint64_t)buffer->data[buffer->cursor++]<<(index*8);return value;}
static void put_double(Buffer *buffer,double value){uint64_t bits;memcpy(&bits,&value,8);put_unsigned(buffer,bits,8);}
static double take_double(Buffer *buffer){uint64_t bits=take_unsigned(buffer,8);double value;memcpy(&value,&bits,8);return value;}
static FILE *open_state_file(const char *path,const char *mode){
#ifdef _WIN32
    int count=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path,-1,NULL,0);if(!count)hv_fail("状态文件路径不是有效 UTF-8");wchar_t *wide=allocate(checked_size((size_t)count,sizeof(wchar_t)));if(!MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path,-1,wide,count))hv_fail("状态文件路径转换失败");FILE *file=_wfopen(wide,!strcmp(mode,"wb")?L"wb":L"rb");release(wide);return file;
#else
    return fopen(path,mode);
#endif
}
static void state_save(HValue path,HValue tensors,HValue integers,HValue floats){require(tensors,H_LIST);require(integers,H_LIST);require(floats,H_LIST);Buffer payload={0};put_unsigned(&payload,(uint64_t)hv_length(tensors),8);put_unsigned(&payload,(uint64_t)hv_length(integers),8);put_unsigned(&payload,(uint64_t)hv_length(floats),8);for(int64_t index=0;index<hv_length(tensors);index++){HObject *tensor=require(hv_index(tensors,hv_i64(index)),H_TENSOR);int dtype=tensor_dtype(tensor);put_unsigned(&payload,(uint64_t)dtype,4);put_unsigned(&payload,(uint64_t)tensor->as.tensor.rank,4);for(int64_t axis=0;axis<tensor->as.tensor.rank;axis++)put_unsigned(&payload,(uint64_t)tensor->as.tensor.shape[axis],8);size_t width=dtype_size(dtype);for(int64_t item=0;item<tensor->as.tensor.count;item++){uint64_t bits=0;int64_t offset=tensor_offset(tensor,item);void *data=tensor->as.tensor.storage->as.storage.data;if(dtype==D_I64)memcpy(&bits,(int64_t*)data+offset,8);else if(dtype==D_F64)memcpy(&bits,(double*)data+offset,8);else if(dtype==D_F32){uint32_t small;memcpy(&small,(float*)data+offset,4);bits=small;}else bits=((unsigned char*)data)[offset];put_unsigned(&payload,bits,width);}}
    for(int64_t index=0;index<hv_length(integers);index++){int64_t value=list_int(integers,index);uint64_t bits;memcpy(&bits,&value,8);put_unsigned(&payload,bits,8);}for(int64_t index=0;index<hv_length(floats);index++)put_double(&payload,hv_number(hv_index(floats,hv_i64(index))));Buffer header={0};buffer_append(&header,"HENGST01",8);put_unsigned(&header,1,4);put_unsigned(&header,payload.size,8);put_unsigned(&header,checksum(payload.data,payload.size),8);FILE *file=open_state_file(text_of(path),"wb");if(!file)hv_fail("无法写入状态文件");int failed=fwrite(header.data,1,header.size,file)!=header.size||fwrite(payload.data,1,payload.size,file)!=payload.size;if(fclose(file)!=0)failed=1;release(header.data);release(payload.data);if(failed)hv_fail("状态文件写入失败");}
static size_t state_file_size(FILE *file){
#ifdef _WIN32
    if(_fseeki64(file,0,SEEK_END)!=0)hv_fail("无法获取状态文件大小");int64_t length=_ftelli64(file);if(length<28||(uint64_t)length>SIZE_MAX||_fseeki64(file,0,SEEK_SET)!=0)hv_fail("状态文件头或大小无效");return (size_t)length;
#else
    if(fseek(file,0,SEEK_END)!=0)hv_fail("无法获取状态文件大小");long length=ftell(file);if(length<28||(uint64_t)length>SIZE_MAX||fseek(file,0,SEEK_SET)!=0)hv_fail("状态文件头或大小无效");return (size_t)length;
#endif
}
static HValue state_load(HValue path,int selection){FILE *file=open_state_file(text_of(path),"rb");if(!file)hv_fail("无法读取状态文件");Buffer input={0};input.size=state_file_size(file);input.data=allocate(input.size);if(fread(input.data,1,input.size,file)!=input.size){fclose(file);hv_fail("状态文件读取失败");}if(fclose(file)!=0)hv_fail("状态文件关闭失败");if(memcmp(input.data,"HENGST01",8))hv_fail("状态文件 magic 无效");input.cursor=8;if(take_unsigned(&input,4)!=1)hv_fail("状态文件版本不支持");uint64_t payload_size=take_unsigned(&input,8),hash=take_unsigned(&input,8);if(payload_size!=input.size-28||checksum(input.data+28,input.size-28)!=hash)hv_fail("状态文件尺寸或校验和错误");uint64_t tensors=take_unsigned(&input,8),integers=take_unsigned(&input,8),floats=take_unsigned(&input,8);if(tensors>INT64_MAX||integers>INT64_MAX||floats>INT64_MAX||tensors>(input.size-input.cursor)/8||integers>(input.size-input.cursor)/8||floats>(input.size-input.cursor)/8)hv_fail("状态文件数量越界");uint64_t selected=selection==0?tensors:selection==1?integers:floats;HValue result=hv_list(0,NULL);HObject *list=result.as.obj;list->as.list.items=resize(list->as.list.items,checked_size((size_t)selected,sizeof(HValue)));list->as.list.capacity=(size_t)selected;
    for(uint64_t index=0;index<tensors;index++){int dtype=(int)take_unsigned(&input,4);uint64_t rank=take_unsigned(&input,4);if(dtype<D_F64||dtype>D_BOOL||rank>(input.size-input.cursor)/8)hv_fail("状态张量元数据无效");int64_t *shape=allocate(checked_size((size_t)rank,sizeof(int64_t)));for(uint64_t axis=0;axis<rank;axis++){uint64_t dimension=take_unsigned(&input,8);if(dimension>INT64_MAX)hv_fail("状态张量维度越界");shape[axis]=(int64_t)dimension;}int64_t count=shape_count((int64_t)rank,shape);size_t width=dtype_size(dtype),bytes=checked_size((size_t)count,width);if(bytes>input.size-input.cursor)hv_fail("状态张量数据被截断");if(selection==0){HValue value=tensor_new((int64_t)rank,shape,dtype);void *data=value.as.obj->as.tensor.storage->as.storage.data;for(int64_t item=0;item<count;item++){uint64_t bits=take_unsigned(&input,width);if(dtype==D_F32){uint32_t small=(uint32_t)bits;memcpy((float*)data+item,&small,4);}else if(dtype==D_BOOL){if(bits>1)hv_fail("状态 bool 数据无效");((unsigned char*)data)[item]=(unsigned char)bits;}else memcpy((char*)data+(size_t)item*width,&bits,width);}list->as.list.items[list->as.list.count++]=value;}else{if(dtype==D_BOOL)for(size_t item=0;item<bytes;item++)if(input.data[input.cursor+item]>1)hv_fail("状态 bool 数据无效");input.cursor+=bytes;}release(shape);}
    if(integers>(input.size-input.cursor)/8)hv_fail("状态整数数据被截断");for(uint64_t index=0;index<integers;index++){uint64_t bits=take_unsigned(&input,8);int64_t value;memcpy(&value,&bits,8);if(selection==1)list->as.list.items[list->as.list.count++]=hv_i64(value);}if(floats>(input.size-input.cursor)/8)hv_fail("状态浮点数据被截断");for(uint64_t index=0;index<floats;index++){double value=take_double(&input);if(selection==2)list->as.list.items[list->as.list.count++]=hv_f64(value);}if(input.cursor!=input.size)hv_fail("状态文件存在额外数据");release(input.data);return result;}
#define FLOAT_UNARY_LOOP(expression) do {for(int64_t index=0;index<count;index++){double value=first[index];target[index]=(expression);}} while(0)
#define FLOAT_UNARY_TYPES(input_type,output_type) do { \
    const input_type *first=(const input_type*)source->as.tensor.storage->as.storage.data+source->as.tensor.offset; \
    output_type *target=output->as.tensor.storage->as.storage.data; \
    if(!strcmp(name,"scale"))FLOAT_UNARY_LOOP(value*parameter); \
    else if(!strcmp(name,"relu"))FLOAT_UNARY_LOOP(isnan(value)?value:value>0?value:0); \
    else if(!strcmp(name,"exp"))FLOAT_UNARY_LOOP(exp(value)); \
    else if(!strcmp(name,"log"))FLOAT_UNARY_LOOP(log(value)); \
    else if(!strcmp(name,"tanh"))FLOAT_UNARY_LOOP(tanh(value)); \
    else if(!strcmp(name,"sqrt"))FLOAT_UNARY_LOOP(sqrt(value)); \
    else {for(int64_t index=0;index<count;index++){double scaled=(double)first[index]*parameter;if(dtype==D_F32)scaled=(float)scaled;double affine=scaled+bias;target[index]=isnan(affine)?affine:affine>0?affine:0;}} \
} while(0)
static int tensor_unary_float(const char *name,HObject *source,HObject *output,double parameter,double bias){
    int dtype=tensor_dtype(source);
    if((dtype!=D_F64&&dtype!=D_F32)||!tensor_contiguous(source))return 0;
    if(strcmp(name,"scale")&&strcmp(name,"relu")&&strcmp(name,"exp")&&strcmp(name,"log")&&
       strcmp(name,"tanh")&&strcmp(name,"sqrt")&&strcmp(name,"fused_affine_relu"))return 0;
    int64_t count=source->as.tensor.count;
    if(!count)return 1;
    if(dtype==D_F64)FLOAT_UNARY_TYPES(double,double);
    else if(tensor_dtype(output)==D_F32)FLOAT_UNARY_TYPES(float,float);
    else FLOAT_UNARY_TYPES(float,double);
    return 1;
}
#undef FLOAT_UNARY_TYPES
#undef FLOAT_UNARY_LOOP

#ifdef __clang__
#pragma STDC FP_CONTRACT OFF
#endif
static HObject *text_list_tensor(HObject *list,int64_t index){
    if(index<0||index>=(int64_t)list->as.list.count)hv_fail("文本模型参数列表索引越界");
    return require(list->as.list.items[index],H_TENSOR);
}
static int text_tensor_shape(HObject *tensor,int64_t rank,const int64_t *shape){
    if(tensor_dtype(tensor)!=D_F64||tensor->as.tensor.rank!=rank||!tensor_contiguous(tensor))return 0;
    for(int64_t axis=0;axis<rank;axis++)if(tensor->as.tensor.shape[axis]!=shape[axis])return 0;
    return 1;
}
__attribute__((target("avx,fma")))
static HValue text_adam_step_f64(HValue *arguments){
    HObject *parameters=require(arguments[0],H_LIST);
    HObject *first_values=require(arguments[1],H_LIST);
    HObject *second_values=require(arguments[2],H_LIST);
    HObject *input=require(arguments[3],H_TENSOR),*target=require(arguments[4],H_TENSOR);
    if(parameters->as.list.count!=4||first_values->as.list.count!=4||second_values->as.list.count!=4)return hv_list(0,NULL);
    if(tensor_dtype(input)!=D_F64||tensor_dtype(target)!=D_F64||input->as.tensor.rank!=2||target->as.tensor.rank!=1||!tensor_contiguous(input)||!tensor_contiguous(target))return hv_list(0,NULL);
    int64_t batch=input->as.tensor.shape[0],input_width=input->as.tensor.shape[1];
    int64_t hidden_width=text_list_tensor(parameters,1)->as.tensor.shape[0];
    int64_t vocabulary=text_list_tensor(parameters,3)->as.tensor.shape[0];
    int64_t w1_shape[2]={input_width,hidden_width},b_shape[1]={hidden_width},w2_shape[2]={hidden_width,vocabulary},o_shape[1]={vocabulary};
    HObject *w1=text_list_tensor(parameters,0),*b1=text_list_tensor(parameters,1),*w2=text_list_tensor(parameters,2),*b2=text_list_tensor(parameters,3);
    if(!text_tensor_shape(w1,2,w1_shape)||!text_tensor_shape(b1,1,b_shape)||!text_tensor_shape(w2,2,w2_shape)||!text_tensor_shape(b2,1,o_shape)||target->as.tensor.shape[0]!=batch||batch<=0)return hv_list(0,NULL);
    const double *input_data=input->as.tensor.storage->as.storage.data,*target_data=target->as.tensor.storage->as.storage.data;
    const double *w1_data=w1->as.tensor.storage->as.storage.data,*b1_data=b1->as.tensor.storage->as.storage.data,*w2_data=w2->as.tensor.storage->as.storage.data,*b2_data=b2->as.tensor.storage->as.storage.data;
    HValue gradients[4]={tensor_allocate(2,w1_shape,D_F64,1),tensor_allocate(1,b_shape,D_F64,1),tensor_allocate(2,w2_shape,D_F64,1),tensor_allocate(1,o_shape,D_F64,1)};
    double *g1=gradients[0].as.obj->as.tensor.storage->as.storage.data,*gb1=gradients[1].as.obj->as.tensor.storage->as.storage.data,*g2=gradients[2].as.obj->as.tensor.storage->as.storage.data,*gb2=gradients[3].as.obj->as.tensor.storage->as.storage.data;
    double *hidden=allocate(checked_size((size_t)hidden_width,sizeof(double))),*logits=allocate(checked_size((size_t)vocabulary,sizeof(double))),*delta=allocate(checked_size((size_t)vocabulary,sizeof(double))),*hidden_delta=allocate(checked_size((size_t)hidden_width,sizeof(double)));
    for(int64_t row=0;row<batch;row++){
        const double *input_row=input_data+row*input_width;
        for(int64_t h=0;h<hidden_width;h++)hidden[h]=b1_data[h];
#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
        if(matmul_has_fma()){
            for(int64_t i=0;i<input_width;i++){
                __m256d input_value=_mm256_set1_pd(input_row[i]);
                int64_t h=0;
                for(;hidden_width-h>=4;h+=4){
                    __m256d accum=_mm256_loadu_pd(hidden+h),weights=_mm256_loadu_pd(w1_data+i*hidden_width+h);
                    _mm256_storeu_pd(hidden+h,_mm256_fmadd_pd(input_value,weights,accum));
                }
                for(;h<hidden_width;h++)hidden[h]+=input_row[i]*w1_data[i*hidden_width+h];
            }
        }else
#endif
        for(int64_t h=0;h<hidden_width;h++){
            double total=b1_data[h];
            for(int64_t i=0;i<input_width;i++)total+=input_row[i]*w1_data[i*hidden_width+h];
            hidden[h]=total;
        }
        for(int64_t h=0;h<hidden_width;h++)hidden[h]=tanh(hidden[h]);
        for(int64_t v=0;v<vocabulary;v++)logits[v]=b2_data[v];
#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
        if(matmul_has_fma()){
            for(int64_t h=0;h<hidden_width;h++){
                __m256d hidden_value=_mm256_set1_pd(hidden[h]);
                int64_t v=0;
                for(;vocabulary-v>=4;v+=4){
                    __m256d accum=_mm256_loadu_pd(logits+v),weights=_mm256_loadu_pd(w2_data+h*vocabulary+v);
                    _mm256_storeu_pd(logits+v,_mm256_fmadd_pd(hidden_value,weights,accum));
                }
                for(;v<vocabulary;v++)logits[v]+=hidden[h]*w2_data[h*vocabulary+v];
            }
        }else
#endif
        for(int64_t v=0;v<vocabulary;v++){
            double total=b2_data[v];
            for(int64_t h=0;h<hidden_width;h++)total+=hidden[h]*w2_data[h*vocabulary+v];
            logits[v]=total;
        }
        double maximum=-INFINITY;
        for(int64_t v=0;v<vocabulary;v++)if(logits[v]>maximum)maximum=logits[v];
        double normalization=0;for(int64_t v=0;v<vocabulary;v++){delta[v]=exp(logits[v]-maximum);normalization+=delta[v];}
        int64_t label=(int64_t)target_data[row];if(label<0||label>=vocabulary)hv_fail("文本模型标签越界");
        for(int64_t v=0;v<vocabulary;v++){delta[v]=delta[v]/normalization-(v==label);gb2[v]+=delta[v];}
        for(int64_t h=0;h<hidden_width;h++){
            double total=0;
#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
            if(matmul_has_fma()){
                int64_t v=0;
                __m256d hidden_value=_mm256_set1_pd(hidden[h]);
                __m256d sum=_mm256_setzero_pd();
                for(;vocabulary-v>=4;v+=4){
                    __m256d delta_values=_mm256_loadu_pd(delta+v),weights=_mm256_loadu_pd(w2_data+h*vocabulary+v);
                    sum=_mm256_add_pd(sum,_mm256_mul_pd(delta_values,weights));
                    _mm256_storeu_pd(g2+h*vocabulary+v,_mm256_add_pd(_mm256_loadu_pd(g2+h*vocabulary+v),_mm256_mul_pd(hidden_value,delta_values)));
                }
                double partial[4];_mm256_storeu_pd(partial,sum);total=partial[0]+partial[1]+partial[2]+partial[3];
                for(;v<vocabulary;v++){g2[h*vocabulary+v]+=hidden[h]*delta[v];total+=delta[v]*w2_data[h*vocabulary+v];}
            }else
#endif
            {for(int64_t v=0;v<vocabulary;v++){g2[h*vocabulary+v]+=hidden[h]*delta[v];total+=delta[v]*w2_data[h*vocabulary+v];}}
            hidden_delta[h]=total*(1.0-hidden[h]*hidden[h]);gb1[h]+=hidden_delta[h];
        }
        for(int64_t i=0;i<input_width;i++){
#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
            if(matmul_has_fma()){
                __m256d input_value=_mm256_set1_pd(input_row[i]);int64_t h=0;
                for(;hidden_width-h>=4;h+=4){__m256d gradient=_mm256_loadu_pd(g1+i*hidden_width+h),hidden_gradient=_mm256_loadu_pd(hidden_delta+h);_mm256_storeu_pd(g1+i*hidden_width+h,_mm256_add_pd(gradient,_mm256_mul_pd(input_value,hidden_gradient)));}
                for(;h<hidden_width;h++)g1[i*hidden_width+h]+=input_row[i]*hidden_delta[h];
            }else
#endif
            for(int64_t h=0;h<hidden_width;h++)g1[i*hidden_width+h]+=input_row[i]*hidden_delta[h];
        }
    }
    release(hidden);release(logits);release(delta);release(hidden_delta);
    double beta1=hv_number(arguments[5]),beta2=hv_number(arguments[6]),first_correction=hv_number(arguments[7]),second_correction=hv_number(arguments[8]),epsilon=hv_number(arguments[9]),rate=hv_number(arguments[10]),decay=hv_number(arguments[11]);
    const double *gradient_data[4]={g1,gb1,g2,gb2};
    HValue output[4];
    for(int index=0;index<4;index++){
        HObject *parameter=text_list_tensor(parameters,index),*moment_first=text_list_tensor(first_values,index),*moment_second=text_list_tensor(second_values,index);
        HValue next=tensor_allocate(parameter->as.tensor.rank,parameter->as.tensor.shape,D_F64,0),next_first=tensor_allocate(parameter->as.tensor.rank,parameter->as.tensor.shape,D_F64,0),next_second=tensor_allocate(parameter->as.tensor.rank,parameter->as.tensor.shape,D_F64,0);
        const double *weights=parameter->as.tensor.storage->as.storage.data,*old_first=moment_first->as.tensor.storage->as.storage.data,*old_second=moment_second->as.tensor.storage->as.storage.data;double *weights_next=next.as.obj->as.tensor.storage->as.storage.data,*first_next=next_first.as.obj->as.tensor.storage->as.storage.data,*second_next=next_second.as.obj->as.tensor.storage->as.storage.data;
        for(int64_t item=0;item<parameter->as.tensor.count;item++){double grad=gradient_data[index][item]/(double)batch,first_value=old_first[item]*beta1+grad*(1.0-beta1),second_value=old_second[item]*beta2+grad*grad*(1.0-beta2),update=first_value*first_correction/(sqrt(second_value*second_correction)+epsilon),updated=weights[item]-rate*update;weights_next[item]=decay>0?updated-rate*decay*weights[item]:updated;first_next[item]=first_value;second_next[item]=second_value;}
        first_values->as.list.items[index]=next_first;second_values->as.list.items[index]=next_second;output[index]=next;
    }
    return hv_list(4,output);
}

static HValue tensor_adam_f64(HValue *arguments){
    HObject *value=require(arguments[0],H_TENSOR);
    const double *inputs[4];
    size_t shape_bytes=(size_t)value->as.tensor.rank*sizeof(int64_t);
    if(!optimized)return hv_list(0,NULL);
    for(int index=0;index<4;index++){
        HObject *tensor=require(arguments[index],H_TENSOR);
        if(tensor_dtype(tensor)!=D_F64||tensor->as.tensor.rank!=value->as.tensor.rank||
           memcmp(tensor->as.tensor.shape,value->as.tensor.shape,shape_bytes)||!tensor_contiguous(tensor))return hv_list(0,NULL);
        inputs[index]=(const double*)tensor->as.tensor.storage->as.storage.data+tensor->as.tensor.offset;
    }
    double beta1=hv_number(arguments[4]),beta2=hv_number(arguments[5]);
    double first_correction=hv_number(arguments[6]),second_correction=hv_number(arguments[7]);
    double epsilon=hv_number(arguments[8]),rate=hv_number(arguments[9]),decay=hv_number(arguments[10]);
    if(!(epsilon>0))return hv_list(0,NULL);
    HValue output[3];
    double *destinations[3];
    for(int index=0;index<3;index++){
        output[index]=tensor_allocate(value->as.tensor.rank,value->as.tensor.shape,D_F64,0);
        destinations[index]=output[index].as.obj->as.tensor.storage->as.storage.data;
    }
    const double *weights=inputs[0],*gradients=inputs[1],*first_input=inputs[2],*second_input=inputs[3];
    double *weights_next=destinations[0],*first_next=destinations[1],*second_next=destinations[2];
    double first_weight=1.0-beta1,second_weight=1.0-beta2;
    for(int64_t index=0;index<value->as.tensor.count;index++){
        double grad=gradients[index];
        double first=first_input[index]*beta1+grad*first_weight;
        double second=second_input[index]*beta2+(grad*grad)*second_weight;
        double update=(first*first_correction)/(sqrt(second*second_correction)+epsilon);
        double next=weights[index]-update*rate;
        weights_next[index]=decay>0?next-weights[index]*(rate*decay):next;
        first_next[index]=first;
        second_next[index]=second;
    }
    return hv_list(3,output);
}
#ifdef __clang__
#pragma STDC FP_CONTRACT DEFAULT
#endif
static HValue native_dispatch(const char *name,int64_t count,HValue *arguments);
typedef struct BuiltinSignature { const char *name; int count; } BuiltinSignature;
static const BuiltinSignature signatures[]={
    {"print",1},{"assert",2},{"assert_close",4},{"len",1},{"push",2},{"pop",1},{"clear",1},{"str",1},{"i64",1},{"f64",1},{"f32",1},{"result_ok",1},{"result_err",1},{"result_is_ok",1},{"result_value",1},{"result_error",1},
    {"tensor",2},{"zeros",1},{"ones",1},{"full",2},{"arange",1},{"rand",1},{"randn",1},{"seed",1},{"rng_state",0},{"rng_restore",1},{"shape",1},{"rank",1},{"numel",1},{"dtype",1},{"cast",2},{"item",2},{"get",2},{"set",3},{"tensor_version",1},{"clone",1},
    {"add",2},{"sub",2},{"mul",2},{"div",2},{"scale",2},{"pow",2},{"exp",1},{"log",1},{"sqrt",1},{"tanh",1},{"sigmoid",1},{"relu",1},{"abs",1},{"sign",1},{"greater",2},{"equal",2},
    {"reshape",2},{"transpose",2},{"slice",5},{"concat",3},{"gather",3},{"scatter_add",4},{"broadcast_to",2},{"reduce_to_shape",2},{"sum",3},{"sum_all",1},{"mean",3},{"max",3},{"argmax",2},{"matmul",2},{"softmax",2},{"unfold2d",5},{"fold2d",6},
    {"coo",5},{"csr",5},{"sparse_dense",1},{"sparse_matmul",2},{"sparse_transpose",1},{"set_threads",1},{"get_threads",0},{"set_optimized",1},{"set_matmul_fast",1},{"adam_f64",11},{"text_adam_step_f64",12},{"fused_affine_relu",3},{"gc_collect",0},{"live_objects",0},{"memory_bytes",0},{"peak_memory_bytes",0},{"allocation_count",0},{"buffer_reuse_count",0},{"cached_memory_bytes",0},{"system_allocation_count",0},{"trim_buffers",0},{"clock",0},{"state_save",4},{"state_tensors",1},{"state_ints",1},{"state_floats",1}
};
HValue hv_native(const char *name,int64_t count,HValue *arguments){if(!name||count<0||(count&&!arguments))hv_fail("内置调用参数无效");int found=0;for(size_t index=0;index<sizeof(signatures)/sizeof(signatures[0]);index++)if(!strcmp(name,signatures[index].name)){found=1;if(count!=signatures[index].count)hv_fail("内置参数数量不匹配");break;}if(!found)hv_fail("未知内置函数");HFrame frame;hv_frame_push(&frame,arguments,(size_t)count);HValue result=native_dispatch(name,count,arguments);hv_frame_pop(&frame);return result;}
static HValue native_dispatch(const char *name,int64_t count,HValue *arguments){
    (void)count;
    if(!strcmp(name,"print")||!strcmp(name,"str")){Buffer buffer={0};format_value(&buffer,arguments[0],0);buffer_append(&buffer,"",1);if(!strcmp(name,"str")){HValue result=hv_string((char*)buffer.data);release(buffer.data);return result;}fputs((char*)buffer.data,stdout);fputc('\n',stdout);release(buffer.data);return hv_void();}
    if(!strcmp(name,"assert")){if(!hv_truth(arguments[0]))hv_fail(text_of(arguments[1]));return hv_void();}
    if(!strcmp(name,"assert_close")){HObject *left=require(arguments[0],H_TENSOR),*right=require(arguments[1],H_TENSOR);double absolute=hv_number(arguments[2]),relative=hv_number(arguments[3]);if(!isfinite(absolute)||!isfinite(relative)||absolute<0||relative<0)hv_fail("assert_close 容差无效");if(left->as.tensor.rank!=right->as.tensor.rank)hv_fail("assert_close rank 不匹配");for(int64_t axis=0;axis<left->as.tensor.rank;axis++)if(left->as.tensor.shape[axis]!=right->as.tensor.shape[axis])hv_fail("assert_close shape 不匹配");for(int64_t index=0;index<left->as.tensor.count;index++){double a=tensor_read(left,index),b=tensor_read(right,index);if(a==b)continue;if(!isfinite(a)||!isfinite(b)||fabs(a-b)>absolute+relative*fabs(b))hv_fail("assert_close 数值不匹配");}return hv_void();}
    if(!strcmp(name,"len"))return hv_i64(hv_length(arguments[0]));
    if(!strcmp(name,"push")){HObject *list=require(arguments[0],H_LIST);if(list->as.list.count==INT64_MAX)hv_fail("列表过长");if(list->as.list.count==list->as.list.capacity){size_t capacity=list->as.list.capacity?list->as.list.capacity:4;if(capacity>SIZE_MAX/2)hv_fail("列表容量溢出");capacity*=2;list->as.list.items=resize(list->as.list.items,checked_size(capacity,sizeof(HValue)));list->as.list.capacity=capacity;}list->as.list.items[list->as.list.count++]=arguments[1];return hv_void();}
    if(!strcmp(name,"pop")){HObject *list=require(arguments[0],H_LIST);if(!list->as.list.count)hv_fail("空列表不能 pop");HValue value=list->as.list.items[--list->as.list.count];list->as.list.items[list->as.list.count]=hv_void();return value;}
    if(!strcmp(name,"clear")){HObject *list=require(arguments[0],H_LIST);list->as.list.count=0;return hv_void();}
    if(!strcmp(name,"i64")||!strcmp(name,"f64")||!strcmp(name,"f32")){HValue value=arguments[0];double number;if(value.tag==H_STRING){char *end;errno=0;if(!strcmp(name,"i64")){long long integer=strtoll(text_of(value),&end,10);if(errno||end==text_of(value)||*end)hv_fail("字符串不是有效 i64");return hv_i64((int64_t)integer);}number=strtod(text_of(value),&end);if(errno||end==text_of(value)||*end)hv_fail("字符串不是有效浮点数");}else if(value.tag==H_BOOL)number=(double)value.as.i;else{if(!strcmp(name,"i64")&&value.tag==H_I64)return value;number=hv_number(value);}if(!strcmp(name,"i64"))return hv_i64(number_to_int(number));return !strcmp(name,"f32")?hv_f32(number):hv_f64(number);}
    if(!strcmp(name,"result_ok")||!strcmp(name,"result_err")){HValue result=object_value(H_RESULT);result.as.obj->as.result.ok=!strcmp(name,"result_ok");if(result.as.obj->as.result.ok)result.as.obj->as.result.value=arguments[0];else{require(arguments[0],H_STRING);result.as.obj->as.result.error=arguments[0];}return result;}
    if(!strcmp(name,"result_is_ok"))return hv_bool(require(arguments[0],H_RESULT)->as.result.ok);
    if(!strcmp(name,"result_value")){HObject *result=require(arguments[0],H_RESULT);if(!result->as.result.ok)hv_fail("从错误 Result 读取 value");return result->as.result.value;}
    if(!strcmp(name,"result_error")){HObject *result=require(arguments[0],H_RESULT);if(result->as.result.ok)hv_fail("从成功 Result 读取 error");return result->as.result.error;}
    if(!strcmp(name,"gc_collect")){collect();return hv_void();}
    if(!strcmp(name,"live_objects"))return hv_i64((int64_t)live_count);
    if(!strcmp(name,"memory_bytes"))return hv_i64((int64_t)used_bytes);
    if(!strcmp(name,"peak_memory_bytes"))return hv_i64((int64_t)peak_bytes);
    if(!strcmp(name,"allocation_count"))return hv_i64((int64_t)allocations);
    if(!strcmp(name,"buffer_reuse_count"))return hv_i64((int64_t)storage_reuses);
    if(!strcmp(name,"cached_memory_bytes"))return hv_i64((int64_t)cache_bytes);
    if(!strcmp(name,"system_allocation_count"))return hv_i64((int64_t)system_allocations);
    if(!strcmp(name,"trim_buffers")){trim_storage_cache();return hv_void();}
    if(!strcmp(name,"clock")){
#ifdef _WIN32
        LARGE_INTEGER counter,frequency;if(!QueryPerformanceCounter(&counter)||!QueryPerformanceFrequency(&frequency))hv_fail("高精度时钟不可用");return hv_f64((double)counter.QuadPart/(double)frequency.QuadPart);
#else
        struct timespec now;if(timespec_get(&now,TIME_UTC)!=TIME_UTC)hv_fail("时钟不可用");return hv_f64((double)now.tv_sec+(double)now.tv_nsec*1e-9);
#endif
    }
    if(!strcmp(name,"set_threads")){int64_t threads=hv_integer(arguments[0]);if(threads<1||threads>256)hv_fail("线程数必须在 1..256");thread_count=(int)threads;return hv_void();}
    if(!strcmp(name,"get_threads"))return hv_i64(thread_count);
    if(!strcmp(name,"set_optimized")){optimized=hv_truth(arguments[0]);return hv_void();}
    if(!strcmp(name,"set_matmul_fast")){matmul_fast=hv_truth(arguments[0]);return hv_void();}
    if(!strcmp(name,"adam_f64"))return tensor_adam_f64(arguments);
    if(!strcmp(name,"text_adam_step_f64"))return text_adam_step_f64(arguments);
    if(!strcmp(name,"seed")){int64_t seed=hv_integer(arguments[0]);memcpy(&rng,&seed,8);if(!rng)rng=UINT64_C(88172645463325252);return hv_void();}
    if(!strcmp(name,"rng_state")){int64_t state;memcpy(&state,&rng,8);return hv_i64(state);}
    if(!strcmp(name,"rng_restore")){int64_t state=hv_integer(arguments[0]);if(!state)hv_fail("随机数状态不能为零");memcpy(&rng,&state,8);return hv_void();}
    if(!strcmp(name,"state_save")){state_save(arguments[0],arguments[1],arguments[2],arguments[3]);return hv_void();}
    if(!strcmp(name,"state_tensors")||!strcmp(name,"state_ints")||!strcmp(name,"state_floats"))return state_load(arguments[0],!strcmp(name,"state_tensors")?0:!strcmp(name,"state_ints")?1:2);
    if(!strcmp(name,"coo")||!strcmp(name,"csr"))return sparse_construct(!strcmp(name,"csr"),arguments[0],arguments[1],arguments[2],hv_integer(arguments[3]),hv_integer(arguments[4]));
    if(!strcmp(name,"sparse_dense"))return sparse_dense(require(arguments[0],H_SPARSE));
    if(!strcmp(name,"sparse_transpose"))return sparse_transpose(require(arguments[0],H_SPARSE));
    if(!strcmp(name,"sparse_matmul"))return sparse_matmul(require(arguments[0],H_SPARSE),require(arguments[1],H_TENSOR));
    if(!strcmp(name,"tensor")){HValue result=tensor_from_shape(arguments[1],D_F64);require(arguments[0],H_LIST);if(hv_length(arguments[0])!=result.as.obj->as.tensor.count)hv_fail("tensor 元素数量不匹配");for(int64_t index=0;index<result.as.obj->as.tensor.count;index++)tensor_write(result.as.obj,index,hv_number(hv_index(arguments[0],hv_i64(index))));return result;}
    if(!strcmp(name,"zeros"))return tensor_from_shape(arguments[0],D_F64);
    if(!strcmp(name,"ones")||!strcmp(name,"full")||!strcmp(name,"rand")||!strcmp(name,"randn")){
        HValue result=tensor_from_shape(arguments[0],D_F64);
        int64_t length=result.as.obj->as.tensor.count;
        double *data=result.as.obj->as.tensor.storage->as.storage.data;
        if(!strcmp(name,"rand")){
            for(int64_t index=0;index<length;index++)data[index]=random_unit();
        }else if(!strcmp(name,"randn")){
            for(int64_t index=0;index<length;index++){
                double u=1-random_unit(),v=random_unit();
                data[index]=sqrt(-2*log(u))*cos(6.283185307179586476925286766559*v);
            }
        }else{
            double fill=!strcmp(name,"ones")?1:hv_number(arguments[1]);
            for(int64_t index=0;index<length;index++)data[index]=fill;
        }
        return result;
    }
    if(!strcmp(name,"arange")){int64_t size=hv_integer(arguments[0]);if(size<0)hv_fail("arange 长度不能为负");HValue result=tensor_new(1,&size,D_F64);for(int64_t index=0;index<size;index++)tensor_write(result.as.obj,index,(double)index);return result;}
    HObject *tensor=require(arguments[0],H_TENSOR);
    if(!strcmp(name,"shape")){int64_t rank=tensor->as.tensor.rank;HValue *values=allocate(checked_size((size_t)rank,sizeof(HValue)));for(int64_t axis=0;axis<rank;axis++)values[axis]=hv_i64(tensor->as.tensor.shape[axis]);HValue result=hv_list(rank,values);release(values);return result;}
    if(!strcmp(name,"rank"))return hv_i64(tensor->as.tensor.rank);
    if(!strcmp(name,"numel"))return hv_i64(tensor->as.tensor.count);
    if(!strcmp(name,"dtype"))return hv_string(dtype_name(tensor_dtype(tensor)));
    if(!strcmp(name,"tensor_version"))return hv_i64(tensor->as.tensor.storage->as.storage.version);
    if(!strcmp(name,"item"))return hv_f64(tensor_read(tensor,hv_integer(arguments[1])));
    if(!strcmp(name,"get"))return hv_f64(read_offset(tensor,indexed_offset(tensor,arguments[1])));
    if(!strcmp(name,"set")){int64_t offset=indexed_offset(tensor,arguments[1]);double value=hv_number(arguments[2]);if(tensor->as.tensor.storage->as.storage.version==INT64_MAX)hv_fail("张量版本溢出");write_offset(tensor,offset,value);tensor->as.tensor.storage->as.storage.version++;return hv_void();}
    if(!strcmp(name,"clone"))return tensor_clone(tensor);
    if(!strcmp(name,"cast")){int dtype=parse_dtype(text_of(arguments[1]));HValue result=tensor_new(tensor->as.tensor.rank,tensor->as.tensor.shape,dtype);for(int64_t index=0;index<tensor->as.tensor.count;index++)copy_element(result.as.obj,index,tensor,index);return result;}
    if(!strcmp(name,"add")||!strcmp(name,"sub")||!strcmp(name,"mul")||!strcmp(name,"div")||!strcmp(name,"equal"))return tensor_binary(name,tensor,require(arguments[1],H_TENSOR));
    if(!strcmp(name,"reshape"))return tensor_reshape(tensor,arguments[1]);
    if(!strcmp(name,"transpose"))return tensor_transpose(tensor,arguments[1]);
    if(!strcmp(name,"slice"))return tensor_slice(tensor,hv_integer(arguments[1]),hv_integer(arguments[2]),hv_integer(arguments[3]),hv_integer(arguments[4]));
    if(!strcmp(name,"concat"))return tensor_concat(tensor,require(arguments[1],H_TENSOR),hv_integer(arguments[2]));
    if(!strcmp(name,"gather"))return tensor_gather(tensor,arguments[1],hv_integer(arguments[2]));
    if(!strcmp(name,"scatter_add"))return tensor_scatter(tensor,arguments[1],require(arguments[2],H_TENSOR),hv_integer(arguments[3]));
    if(!strcmp(name,"broadcast_to"))return tensor_broadcast(tensor,arguments[1]);
    if(!strcmp(name,"reduce_to_shape"))return tensor_reduce_shape(tensor,arguments[1]);
    if(!strcmp(name,"sum_all"))return tensor_sum_all(tensor);
    if(!strcmp(name,"sum")||!strcmp(name,"mean")||!strcmp(name,"max"))return tensor_reduce(name,tensor,hv_integer(arguments[1]),hv_truth(arguments[2]));
    if(!strcmp(name,"argmax"))return tensor_reduce(name,tensor,hv_integer(arguments[1]),0);
    if(!strcmp(name,"matmul"))return tensor_matmul(tensor,require(arguments[1],H_TENSOR));
    if(!strcmp(name,"softmax"))return tensor_softmax(tensor,hv_integer(arguments[1]));
    if(!strcmp(name,"unfold2d"))return tensor_unfold(tensor,hv_integer(arguments[1]),hv_integer(arguments[2]),hv_integer(arguments[3]),hv_integer(arguments[4]));
    if(!strcmp(name,"fold2d"))return tensor_fold(tensor,arguments[1],hv_integer(arguments[2]),hv_integer(arguments[3]),hv_integer(arguments[4]),hv_integer(arguments[5]));
    int scalar=!strcmp(name,"scale")||!strcmp(name,"pow")||!strcmp(name,"greater")||!strcmp(name,"fused_affine_relu");double parameter=scalar?hv_number(arguments[1]):0,bias=!strcmp(name,"fused_affine_relu")?hv_number(arguments[2]):0;int dtype=!strcmp(name,"greater")?D_BOOL:tensor_dtype(tensor)==D_F32?D_F32:D_F64;int preserving=!strcmp(name,"relu")||!strcmp(name,"abs")||!strcmp(name,"sign");if(preserving)dtype=tensor_dtype(tensor);if(!strcmp(name,"fused_affine_relu"))dtype=D_F64;
    HValue result=tensor_new(tensor->as.tensor.rank,tensor->as.tensor.shape,dtype);if(tensor_unary_float(name,tensor,result.as.obj,parameter,bias))return result;for(int64_t index=0;index<tensor->as.tensor.count;index++){if(preserving&&dtype==D_I64){int64_t value=((int64_t*)tensor->as.tensor.storage->as.storage.data)[tensor_offset(tensor,index)],output;if(!strcmp(name,"relu"))output=value>0?value:0;else if(!strcmp(name,"sign"))output=value>0?1:value<0?-1:0;else{if(value==INT64_MIN)hv_fail("i64 abs 溢出");output=value<0?-value:value;}((int64_t*)result.as.obj->as.tensor.storage->as.storage.data)[index]=output;continue;}double value=tensor_read(tensor,index),output;
        if(!strcmp(name,"scale"))output=value*parameter;else if(!strcmp(name,"pow"))output=pow(value,parameter);else if(!strcmp(name,"exp"))output=exp(value);else if(!strcmp(name,"log"))output=log(value);else if(!strcmp(name,"sqrt"))output=sqrt(value);else if(!strcmp(name,"tanh"))output=tanh(value);else if(!strcmp(name,"sigmoid")){if(value>=0)output=1/(1+exp(-value));else{double exponential=exp(value);output=exponential/(1+exponential);}}else if(!strcmp(name,"relu"))output=isnan(value)?value:value>0?value:0;else if(!strcmp(name,"abs"))output=fabs(value);else if(!strcmp(name,"sign"))output=isnan(value)?value:value>0?1:value<0?-1:0;else if(!strcmp(name,"greater"))output=compare_numeric(numeric_offset(tensor,tensor_offset(tensor,index)),hv_f64(parameter))==1;else if(!strcmp(name,"fused_affine_relu")){double scaled=value*parameter;if(tensor_dtype(tensor)==D_F32)scaled=(float)scaled;double affine=scaled+bias;output=isnan(affine)?affine:affine>0?affine:0;}else{hv_fail("未实现的内置函数");return hv_void();}tensor_write(result.as.obj,index,output);
    }return result;
}
