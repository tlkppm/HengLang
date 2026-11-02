#include "../runtime/heng_runtime.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#define N0(name) hv_native(name,0,NULL)
#define N1(name,a) hv_native(name,1,(HValue[]){a})
#define N2(name,a,b) hv_native(name,2,(HValue[]){a,b})
#define N3(name,a,b,c) hv_native(name,3,(HValue[]){a,b,c})
#define N4(name,a,b,c,d) hv_native(name,4,(HValue[]){a,b,c,d})
#define N5(name,a,b,c,d,e) hv_native(name,5,(HValue[]){a,b,c,d,e})
#define N6(name,a,b,c,d,e,f) hv_native(name,6,(HValue[]){a,b,c,d,e,f})
static HValue slots[128];
static void check(int condition,const char *message){if(!condition){fprintf(stderr,"测试失败: %s\n",message);exit(2);}}
static void near(double value,double expected,const char *message){check(isfinite(value)&&fabs(value-expected)<=1e-10*(1+fabs(expected)),message);}
static HValue ints(int64_t count,const int64_t *numbers){HValue *values=malloc((size_t)(count?count:1)*sizeof(HValue));check(values!=NULL,"测试分配");for(int64_t index=0;index<count;index++)values[index]=hv_i64(numbers[index]);HValue list=hv_list(count,values);free(values);return list;}
static HValue floats(int64_t count,const double *numbers){HValue *values=malloc((size_t)(count?count:1)*sizeof(HValue));check(values!=NULL,"测试分配");for(int64_t index=0;index<count;index++)values[index]=hv_f64(numbers[index]);HValue list=hv_list(count,values);free(values);return list;}
static HValue shape1(int64_t a){return ints(1,(int64_t[]){a});}
static HValue shape2(int64_t a,int64_t b){return ints(2,(int64_t[]){a,b});}
static HValue matrix(void){return N2("tensor",floats(6,(double[]){1,2,3,4,5,6}),shape2(2,3));}
static double item(HValue tensor,int64_t index){return hv_number(N2("item",tensor,hv_i64(index)));}
static int64_t native_int(const char *name,HValue value){return hv_integer(N1(name,value));}
static void close_tensor(HValue left,HValue right){N4("assert_close",left,right,hv_f64(1e-12),hv_f64(1e-12));}
static void core_tests(void){
    check(hv_integer(hv_binary("+",hv_i64(12),hv_i64(30)))==42,"整数加法");check(hv_integer(hv_binary("-",hv_i64(-3),hv_i64(4)))==-7,"整数减法");check(hv_integer(hv_binary("*",hv_i64(-3),hv_i64(-4)))==12,"整数乘法");check(hv_integer(hv_binary("/",hv_i64(10),hv_i64(3)))==3,"整数除法");check(hv_integer(hv_binary("%",hv_i64(-10),hv_i64(3)))==-1,"整数余数");check(hv_truth(hv_binary("<",hv_i64(9007199254740992LL),hv_i64(9007199254740993LL))),"精确整数比较");check(!hv_truth(hv_binary("==",hv_i64(9007199254740993LL),hv_f64(9007199254740992.0))),"混合数值精确比较");check(hv_truth(hv_binary(">",hv_i64(9007199254740993LL),hv_f64(9007199254740992.0))),"混合数值大小比较");near(hv_number(hv_binary("+",hv_f32(.5),hv_f64(1))),1.5,"浮点加法");check(hv_integer(hv_unary("-",hv_i64(3)))==-3,"一元负号");check(hv_truth(hv_unary("not",hv_bool(0))),"布尔取反");check(hv_truth(hv_binary("and",hv_bool(1),hv_bool(1))),"严格布尔运算");
    slots[0]=hv_binary("+",hv_string("中文"),hv_string("打印"));N1("print",slots[0]);slots[1]=hv_list(0,NULL);N2("push",slots[1],hv_i64(8));N2("push",slots[1],hv_i64(9));check(hv_integer(N1("pop",slots[1]))==9,"列表 pop");hv_set_index(slots[1],hv_i64(0),hv_i64(7));check(hv_integer(hv_index(slots[1],hv_i64(0)))==7,"列表读写");N1("clear",slots[1]);check(native_int("len",slots[1])==0,"列表 clear");N2("push",slots[1],slots[1]);slots[2]=hv_record("Node",1,(const char*[]){"next"},(HValue[]){slots[1]});hv_set_field(slots[2],"next",slots[2]);check(hv_truth(hv_binary("==",slots[2],hv_field(slots[2],"next"))),"记录引用比较");check(!hv_truth(hv_binary("==",slots[1],hv_list(0,NULL))),"列表引用比较");check(hv_truth(hv_binary("==",N1("str",slots[2]),hv_string("Node{next: <cycle>}"))),"循环记录安全格式化");slots[3]=N1("result_ok",slots[2]);check(hv_truth(N1("result_is_ok",slots[3])),"Result 成功");check(hv_truth(hv_binary("==",N1("result_value",slots[3]),slots[2])),"Result value");slots[4]=N1("result_err",hv_string("失败信息"));check(!hv_truth(N1("result_is_ok",slots[4])),"Result 失败");check(hv_truth(hv_binary("==",N1("result_error",slots[4]),hv_string("失败信息"))),"Result error");N0("gc_collect");check(hv_truth(hv_binary("==",hv_field(N1("result_value",slots[3]),"next"),slots[2])),"GC 保留有根环");
    check(hv_integer(N1("i64",hv_string("-123")))==-123,"字符串转 i64");near(hv_number(N1("f64",hv_string("1.25"))),1.25,"字符串转 f64");near(hv_number(N1("f32",hv_i64(2))),2,"数值转 f32");check(hv_truth(hv_binary("==",N1("str",hv_i64(42)),hv_string("42"))),"数值格式化");N2("assert",hv_bool(1),hv_string("成功断言"));
    for(size_t index=0;index<128;index++)slots[index]=hv_void();N0("gc_collect");check(hv_integer(N0("live_objects"))==0,"不可达列表与记录环回收");check(hv_integer(N0("memory_bytes"))==0,"GC 释放全部 tracked 内存");check(hv_integer(N0("peak_memory_bytes"))>0,"峰值内存统计");check(hv_integer(N0("allocation_count"))>0,"分配次数统计");check(hv_number(N0("clock"))>0,"时钟");
}
static void tensor_tests(void){
    slots[0]=matrix();check(native_int("rank",slots[0])==2&&native_int("numel",slots[0])==6,"张量形状");check(hv_integer(hv_index(N1("shape",slots[0]),hv_i64(1)))==3,"shape 列表");check(hv_truth(hv_binary("==",N1("dtype",slots[0]),hv_string("f64"))),"默认 dtype");
    slots[1]=N2("transpose",slots[0],ints(2,(int64_t[]){1,0}));near(item(slots[1],1),4,"转置 stride");slots[2]=N5("slice",slots[1],hv_i64(0),hv_i64(2),hv_i64(-1),hv_i64(-1));near(item(slots[2],0),3,"负步长 view");near(item(slots[2],5),4,"负步长末元素");N3("set",slots[2],shape2(0,0),hv_f64(30));near(hv_number(N2("get",slots[0],shape2(0,2))),30,"view 共享存储");check(native_int("tensor_version",slots[0])==1&&native_int("tensor_version",slots[1])==1,"共享版本");slots[3]=N1("clone",slots[2]);N3("set",slots[3],shape2(0,0),hv_f64(3));near(item(slots[2],0),30,"clone 独立存储");slots[4]=N2("reshape",slots[1],shape1(6));near(item(slots[4],1),4,"非连续 reshape");slots[5]=N2("cast",slots[3],hv_string("f32"));check(hv_truth(hv_binary("==",N1("dtype",slots[5]),hv_string("f32"))),"f32 typed storage");slots[6]=N2("cast",slots[3],hv_string("i64"));slots[7]=N2("cast",slots[3],hv_string("bool"));check(hv_truth(hv_binary("==",N1("dtype",slots[6]),hv_string("i64"))),"i64 typed storage");near(item(slots[7],0),1,"bool typed storage");
    slots[0]=matrix();close_tensor(hv_binary("+",slots[0],hv_f64(2)),N2("add",slots[0],N2("full",shape1(1),hv_f64(2))));near(item(hv_binary("-",hv_i64(10),slots[0]),5),4,"标量减张量运算符");near(item(hv_binary("*",slots[0],hv_i64(2)),5),12,"张量乘标量运算符");near(item(hv_binary("/",hv_i64(12),slots[0]),2),4,"标量除张量运算符");near(item(hv_unary("-",slots[0]),5),-6,"张量一元取负");near(item(hv_unary("+",slots[0]),0),1,"张量一元正号");slots[1]=N2("tensor",floats(3,(double[]){10,20,30}),shape1(3));slots[2]=N2("add",slots[0],slots[1]);near(item(slots[2],5),36,"尾轴广播");near(item(N2("sub",slots[2],slots[1]),4),5,"广播减法");near(item(N2("mul",slots[0],slots[1]),3),40,"广播乘法");near(item(N2("div",slots[1],slots[1]),2),1,"广播除法");near(item(N2("equal",slots[0],slots[0]),0),1,"张量相等");
    slots[3]=N2("broadcast_to",slots[1],shape2(2,3));near(item(slots[3],3),10,"broadcast view");slots[4]=N2("reduce_to_shape",slots[3],shape1(3));near(item(slots[4],2),60,"反广播归约");near(item(N3("sum",slots[0],hv_i64(-1),hv_bool(0)),1),15,"sum 负 axis");near(item(N3("mean",slots[0],hv_i64(0),hv_bool(1)),2),4.5,"mean 保留维度");near(item(N3("max",slots[0],hv_i64(1),hv_bool(0)),1),6,"max");near(item(N2("argmax",slots[0],hv_i64(-1)),0),2,"argmax");near(item(N1("sum_all",slots[0]),0),21,"全归约");
    slots[5]=N3("gather",slots[0],ints(3,(int64_t[]){2,0,2}),hv_i64(1));near(item(slots[5],4),4,"整列表 gather");slots[6]=N4("scatter_add",N1("zeros",shape2(2,3)),ints(3,(int64_t[]){2,0,2}),slots[5],hv_i64(1));near(item(slots[6],2),6,"重复索引 scatter_add");near(item(slots[6],5),12,"scatter_add 后行");slots[7]=N3("concat",slots[0],slots[0],hv_i64(0));near(item(slots[7],7),2,"concat");
    slots[8]=N1("zeros",shape2(0,3));slots[9]=N2("add",slots[8],N1("ones",shape2(1,3)));check(native_int("numel",slots[9])==0,"空维广播");slots[10]=N2("broadcast_to",N1("ones",shape2(1,3)),shape2(0,3));check(native_int("numel",slots[10])==0,"广播到空维");near(item(N3("sum",slots[8],hv_i64(0),hv_bool(0)),0),0,"空维 sum");near(item(N1("sum_all",slots[8]),0),0,"空全归约");slots[11]=N2("reshape",N1("arange",hv_i64(6)),ints(2,(int64_t[]){-1,3}));close_tensor(slots[11],N2("tensor",floats(6,(double[]){0,1,2,3,4,5}),shape2(2,3)));
    int64_t dimensions[80],permutation[80];for(int64_t axis=0;axis<80;axis++){dimensions[axis]=1;permutation[axis]=79-axis;}slots[12]=N1("ones",ints(80,dimensions));slots[13]=N2("transpose",slots[12],ints(80,permutation));check(native_int("rank",N2("add",slots[12],slots[13]))==80,"任意 rank 元数据");near(item(slots[13],0),1,"高 rank transpose");
    slots[14]=N1("zeros",ints(3,(int64_t[]){0,INT64_MAX,INT64_MAX}));check(native_int("numel",slots[14])==0,"空维避免无效尺寸乘积");slots[15]=N5("slice",slots[8],hv_i64(0),hv_i64(0),hv_i64(-1),hv_i64(-1));check(native_int("numel",slots[15])==0,"空维负步长切片");slots[16]=N2("reshape",slots[14],ints(3,(int64_t[]){INT64_MAX,INT64_MAX,0}));check(native_int("numel",slots[16])==0,"空维 reshape 不溢出");slots[17]=N2("softmax",slots[14],hv_i64(1));check(native_int("numel",slots[17])==0,"空维 softmax");
}
static void numerical_tests(void){
    slots[20]=N2("tensor",floats(3,(double[]){-1000,0,1000}),shape1(3));slots[21]=N1("sigmoid",slots[20]);near(item(slots[21],0),0,"稳定 sigmoid 负端");near(item(slots[21],1),.5,"稳定 sigmoid 中点");near(item(slots[21],2),1,"稳定 sigmoid 正端");slots[22]=N2("softmax",N2("tensor",floats(3,(double[]){1000,1001,1002}),shape1(3)),hv_i64(0));near(item(N1("sum_all",slots[22]),0),1,"稳定 softmax");near(item(slots[22],2),.6652409557748218,"softmax 数值");
    slots[23]=N2("tensor",floats(4,(double[]){-2,-1,1,2}),shape1(4));near(item(N1("abs",slots[23]),0),2,"abs");near(item(N1("sign",slots[23]),0),-1,"sign");near(item(N1("relu",slots[23]),0),0,"relu");near(item(N2("greater",slots[23],hv_f64(0)),2),1,"greater");near(item(N2("scale",slots[23],hv_f64(2)),3),4,"scale");near(item(N2("pow",slots[23],hv_f64(2)),0),4,"pow");near(item(N1("tanh",slots[23]),3),tanh(2),"tanh");slots[24]=N2("full",shape1(2),hv_f64(4));near(item(N1("sqrt",slots[24]),0),2,"sqrt");close_tensor(N1("log",N1("exp",slots[24])),slots[24]);slots[25]=N3("fused_affine_relu",slots[23],hv_f64(2),hv_f64(1));close_tensor(slots[25],N1("relu",N2("add",N2("scale",slots[23],hv_f64(2)),N2("full",shape1(1),hv_f64(1)))));
    N1("seed",hv_i64(42));slots[26]=N0("rng_state");slots[27]=N1("rand",shape1(16));N1("rng_restore",slots[26]);close_tensor(slots[27],N1("rand",shape1(16)));N1("seed",hv_i64(7));slots[28]=N0("rng_state");slots[29]=N1("randn",shape1(17));N1("rng_restore",slots[28]);close_tensor(slots[29],N1("randn",shape1(17)));
    slots[30]=N2("tensor",floats(6,(double[]){1,2,3,4,5,6}),shape2(2,3));slots[31]=N2("tensor",floats(6,(double[]){1,2,3,4,5,6}),shape2(3,2));slots[32]=N2("matmul",slots[30],slots[31]);near(item(slots[32],0),22,"matmul");near(item(slots[32],3),64,"matmul 尾元素");slots[33]=N2("reshape",slots[30],ints(3,(int64_t[]){1,2,3}));slots[34]=N2("broadcast_to",slots[33],ints(4,(int64_t[]){2,3,2,3}));slots[35]=N2("matmul",slots[34],slots[31]);check(native_int("numel",slots[35])==24,"批维广播 matmul");near(item(slots[35],23),64,"批维 matmul 数值");
    slots[36]=N2("reshape",N1("arange",hv_i64(70)),shape2(2,35));slots[37]=N2("reshape",N1("arange",hv_i64(1400)),shape2(35,40));N1("set_optimized",hv_bool(0));N1("set_threads",hv_i64(1));slots[38]=N2("matmul",slots[36],slots[37]);N1("set_optimized",hv_bool(1));N1("set_threads",hv_i64(2));slots[39]=N2("matmul",slots[36],slots[37]);close_tensor(slots[38],slots[39]);N1("set_threads",hv_i64(1));
    slots[40]=N2("cast",slots[30],hv_string("i64"));slots[41]=N2("cast",slots[31],hv_string("i64"));close_tensor(N2("matmul",slots[40],slots[41]),slots[32]);near(item(N3("mean",slots[40],hv_i64(1),hv_bool(0)),1),5,"整数 mean");check(hv_truth(hv_binary("==",N1("dtype",N2("argmax",slots[40],hv_i64(0))),hv_string("i64"))),"argmax dtype");
    slots[42]=N2("tensor",floats(1,(double[]){9007199254740992.0}),shape1(1));slots[43]=N2("cast",slots[42],hv_string("i64"));slots[44]=N2("cast",N1("ones",shape1(1)),hv_string("i64"));slots[45]=N2("add",slots[43],slots[44]);check(item(N2("equal",slots[43],slots[45]),0)==0,"大整数张量精确比较");check(item(N2("equal",slots[42],slots[45]),0)==0,"混合 dtype 大整数张量比较");check(item(N2("greater",slots[45],hv_f64(9007199254740992.0)),0)==1,"大整数 greater");check(hv_truth(hv_binary("==",N1("dtype",N1("relu",slots[45])),hv_string("i64"))),"整数 relu 保留 dtype");close_tensor(N2("sub",N1("relu",slots[45]),slots[43]),slots[44]);close_tensor(N2("sub",slots[45],slots[43]),slots[44]);
}
static void convolution_sparse_tests(void){
    slots[50]=N2("reshape",N1("arange",hv_i64(9)),ints(4,(int64_t[]){1,1,3,3}));slots[51]=N5("unfold2d",slots[50],hv_i64(2),hv_i64(2),hv_i64(1),hv_i64(0));check(native_int("numel",slots[51])==16,"unfold 大小");near(item(slots[51],0),0,"unfold 首元素");near(item(slots[51],15),8,"unfold 末元素");slots[52]=N6("fold2d",slots[51],ints(4,(int64_t[]){1,1,3,3}),hv_i64(2),hv_i64(2),hv_i64(1),hv_i64(0));near(item(slots[52],4),16,"fold 重叠累加");near(item(slots[52],8),8,"fold 边界");slots[53]=N5("unfold2d",slots[50],hv_i64(3),hv_i64(3),hv_i64(2),hv_i64(1));near(item(slots[53],0),0,"unfold padding");slots[54]=N6("fold2d",slots[53],ints(4,(int64_t[]){1,1,3,3}),hv_i64(3),hv_i64(3),hv_i64(2),hv_i64(1));near(item(slots[54],4),16,"fold stride padding");
    slots[55]=N5("coo",ints(3,(int64_t[]){1,0,1}),ints(3,(int64_t[]){1,0,0}),floats(3,(double[]){4,2,3}),hv_i64(2),hv_i64(2));slots[56]=N1("sparse_dense",slots[55]);near(item(slots[56],3),4,"COO 稠密转换");slots[57]=N5("csr",ints(3,(int64_t[]){0,1,3}),ints(3,(int64_t[]){0,0,1}),floats(3,(double[]){2,3,4}),hv_i64(2),hv_i64(2));close_tensor(N1("sparse_dense",slots[57]),slots[56]);slots[58]=N1("sparse_transpose",slots[55]);close_tensor(N1("sparse_dense",slots[58]),N2("transpose",slots[56],shape2(1,0)));slots[59]=N2("sparse_matmul",slots[55],slots[56]);close_tensor(slots[59],N2("matmul",slots[56],slots[56]));
    slots[60]=N5("coo",shape2(0,0),shape2(0,0),floats(2,(double[]){1,2}),hv_i64(1),hv_i64(1));near(item(N1("sparse_dense",slots[60]),0),3,"COO 重复坐标累加");
    slots[61]=N5("coo",ints(5,(int64_t[]){1,0,1,0,1}),ints(5,(int64_t[]){0,1,0,1,1}),floats(5,(double[]){2,3,5,-1,4}),hv_i64(2),hv_i64(2));
    slots[62]=N2("tensor",floats(4,(double[]){0,2,7,4}),shape2(2,2));close_tensor(N1("sparse_dense",slots[61]),slots[62]);close_tensor(N1("sparse_dense",N1("sparse_transpose",slots[61])),N2("transpose",slots[62],shape2(1,0)));close_tensor(N2("sparse_matmul",slots[61],slots[62]),N2("matmul",slots[62],slots[62]));
}
static void storage_cache_tests(void){
    for(size_t index=0;index<128;index++)slots[index]=hv_void();N0("gc_collect");N0("trim_buffers");
    check(hv_integer(N0("cached_memory_bytes"))==0,"缓冲池初始清空");
    slots[0]=N2("full",shape1(1024),hv_f64(7));slots[1]=N2("reshape",slots[0],shape2(32,32));slots[0]=hv_void();N0("gc_collect");
    int64_t protected_reuses=hv_integer(N0("buffer_reuse_count"));slots[2]=N1("zeros",shape1(1024));check(hv_integer(N0("buffer_reuse_count"))==protected_reuses,"有根视图存储不能提前复用");near(item(slots[1],0),7,"复用不破坏视图别名");near(item(slots[2],0),0,"新存储初始化");
    slots[1]=hv_void();slots[2]=hv_void();N0("gc_collect");
    int64_t before=hv_integer(N0("buffer_reuse_count")),logical=hv_integer(N0("allocation_count")),physical=hv_integer(N0("system_allocation_count"));
    for(int index=0;index<100;index++){slots[0]=N1("zeros",shape1(1024));near(item(slots[0],1023),0,"复用缓冲区清零");slots[0]=hv_void();N0("gc_collect");}
    check(hv_integer(N0("buffer_reuse_count"))-before==100,"临时存储持续复用");check(hv_integer(N0("system_allocation_count"))-physical<hv_integer(N0("allocation_count"))-logical,"复用减少系统分配");
    for(int index=0;index<64;index++){slots[0]=N1("zeros",shape1(16384+index));slots[0]=hv_void();N0("gc_collect");}
    check(hv_integer(N0("cached_memory_bytes"))<=16*1024*1024,"缓冲池内存上限");check(hv_integer(N0("memory_bytes"))==0,"复用压力检查活动内存归零");N0("trim_buffers");check(hv_integer(N0("cached_memory_bytes"))==0,"显式释放缓存内存");
}
static void gc_tests(void){
    for(size_t index=0;index<128;index++)slots[index]=hv_void();N0("gc_collect");check(hv_integer(N0("memory_bytes"))==0,"中途 GC 全释放");
    slots[0]=N2("transpose",matrix(),shape2(1,0));for(int iteration=0;iteration<2000;iteration++){HValue cycle=hv_list(0,NULL);N2("push",cycle,cycle);hv_safepoint();}N0("gc_collect");near(item(slots[0],1),4,"GC 保留仅 view 引用存储");check(hv_integer(N0("live_objects"))==2,"GC view 与 storage 对象数");slots[0]=hv_void();N0("gc_collect");check(hv_integer(N0("memory_bytes"))==0,"循环 GC 内存归零");
    HValue protected_value=hv_string("保护参数");HValue arguments[1]={protected_value};HFrame frame;hv_frame_push(&frame,arguments,1);N0("gc_collect");check(hv_length(arguments[0])>0,"原生嵌套根帧");hv_frame_pop(&frame);N0("gc_collect");
}
static void state_tests(const char *path){
    slots[0]=hv_string(path);slots[1]=matrix();slots[2]=N2("transpose",slots[1],shape2(1,0));slots[3]=N2("cast",slots[2],hv_string("f32"));slots[4]=N2("cast",slots[2],hv_string("i64"));slots[5]=N2("cast",slots[2],hv_string("bool"));slots[6]=N1("zeros",shape2(0,3));slots[7]=N1("sum_all",slots[1]);slots[8]=N5("slice",slots[2],hv_i64(0),hv_i64(2),hv_i64(-1),hv_i64(-1));slots[9]=N2("add",N2("cast",N2("full",shape1(1),hv_f64(9007199254740992.0)),hv_string("i64")),N2("cast",N1("ones",shape1(1)),hv_string("i64")));slots[10]=hv_list(8,(HValue[]){slots[2],slots[3],slots[4],slots[5],slots[6],slots[7],slots[8],slots[9]});slots[11]=ints(3,(int64_t[]){INT64_MIN,0,INT64_MAX});slots[12]=floats(3,(double[]){-0.0,INFINITY,NAN});N4("state_save",slots[0],slots[10],slots[11],slots[12]);slots[13]=N1("state_tensors",slots[0]);slots[14]=N1("state_ints",slots[0]);slots[15]=N1("state_floats",slots[0]);check(hv_length(slots[13])==8,"状态张量数量");for(int64_t index=0;index<8;index++){HValue original=hv_index(slots[10],hv_i64(index)),loaded=hv_index(slots[13],hv_i64(index));close_tensor(original,loaded);check(hv_truth(hv_binary("==",N1("dtype",original),N1("dtype",loaded))),"状态保留 dtype");}check(hv_integer(hv_index(slots[14],hv_i64(0)))==INT64_MIN&&hv_integer(hv_index(slots[14],hv_i64(2)))==INT64_MAX,"状态 i64 极值");check(signbit(hv_number(hv_index(slots[15],hv_i64(0))))&&isinf(hv_number(hv_index(slots[15],hv_i64(1))))&&isnan(hv_number(hv_index(slots[15],hv_i64(2)))),"状态浮点位保留");slots[16]=hv_binary("+",slots[0],hv_string(".copy"));N4("state_save",slots[16],slots[13],slots[14],slots[15]);slots[17]=hv_binary("+",slots[0],hv_string(".中文"));N4("state_save",slots[17],slots[13],slots[14],slots[15]);slots[18]=N1("state_tensors",slots[17]);close_tensor(hv_index(slots[18],hv_i64(0)),slots[2]);N0("gc_collect");close_tensor(hv_index(slots[13],hv_i64(6)),slots[8]);
}
static void failure(const char *mode,const char *path){
    hv_site("native-case.heng",17,9);
    if(!strcmp(mode,"overflow"))hv_binary("+",hv_i64(INT64_MAX),hv_i64(1));
    if(!strcmp(mode,"multiply"))hv_binary("*",hv_i64(INT64_MIN),hv_i64(-1));
    if(!strcmp(mode,"subtract"))hv_binary("-",hv_i64(INT64_MIN),hv_i64(1));
    if(!strcmp(mode,"division"))hv_binary("/",hv_i64(INT64_MIN),hv_i64(-1));
    if(!strcmp(mode,"zero"))hv_binary("/",hv_i64(1),hv_i64(0));
    if(!strcmp(mode,"truth"))hv_truth(hv_i64(1));
    if(!strcmp(mode,"index"))hv_index(hv_list(0,NULL),hv_i64(0));
    if(!strcmp(mode,"pop"))N1("pop",hv_list(0,NULL));
    if(!strcmp(mode,"shape"))N1("zeros",shape1(-1));
    if(!strcmp(mode,"size"))N1("zeros",shape2(INT64_MAX,2));
    if(!strcmp(mode,"axis"))N3("sum",matrix(),hv_i64(2),hv_bool(0));
    if(!strcmp(mode,"slice"))N5("slice",matrix(),hv_i64(1),hv_i64(0),hv_i64(4),hv_i64(1));
    if(!strcmp(mode,"step"))N5("slice",matrix(),hv_i64(1),hv_i64(0),hv_i64(2),hv_i64(0));
    if(!strcmp(mode,"axes"))N2("transpose",matrix(),shape2(0,0));
    if(!strcmp(mode,"broadcast"))N2("add",matrix(),N1("ones",shape1(4)));
    if(!strcmp(mode,"reshape"))N2("reshape",matrix(),shape2(4,2));
    if(!strcmp(mode,"gather"))N3("gather",matrix(),shape1(3),hv_i64(1));
    if(!strcmp(mode,"matmul"))N2("matmul",matrix(),matrix());
    if(!strcmp(mode,"coo_index"))N5("coo",shape1(1),shape1(0),floats(1,(double[]){1}),hv_i64(1),hv_i64(1));
    if(!strcmp(mode,"csr_offsets"))N5("csr",ints(4,(int64_t[]){0,2,1,2}),shape2(0,0),floats(2,(double[]){1,2}),hv_i64(3),hv_i64(1));
    if(!strcmp(mode,"csr_duplicate"))N5("csr",shape2(0,2),shape2(0,0),floats(2,(double[]){1,2}),hv_i64(1),hv_i64(1));
    if(!strcmp(mode,"max_empty"))N3("max",N1("zeros",shape2(2,0)),hv_i64(1),hv_bool(0));
    if(!strcmp(mode,"mean_empty"))N3("mean",N1("zeros",shape2(2,0)),hv_i64(1),hv_bool(0));
    if(!strcmp(mode,"cast_overflow"))N2("cast",N2("full",shape1(1),hv_f64(INFINITY)),hv_string("i64"));
    if(!strcmp(mode,"result"))N1("result_value",N1("result_err",hv_string("错误")));
    if(!strcmp(mode,"arity"))hv_native("zeros",0,NULL);
    if(!strcmp(mode,"unknown"))N0("missing");
    if(!strcmp(mode,"threads"))N1("set_threads",hv_i64(0));
    if(!strcmp(mode,"load"))N1("state_tensors",hv_string(path));
    hv_fail("预期失败分支未触发");
}
int main(int argc,char **argv){HFrame frame;hv_frame_push(&frame,slots,128);if(argc>1&&!strcmp(argv[1],"fail")){failure(argc>2?argv[2]:"unknown",argc>3?argv[3]:"missing.hstate");return 3;}check(argc>1,"需要状态文件路径");core_tests();tensor_tests();numerical_tests();convolution_sparse_tests();storage_cache_tests();gc_tests();state_tests(argv[1]);for(size_t index=0;index<128;index++)slots[index]=hv_void();N0("gc_collect");check(hv_integer(N0("memory_bytes"))==0,"最终内存归零");hv_frame_pop(&frame);hv_shutdown();puts("PASS native runtime");return 0;}
