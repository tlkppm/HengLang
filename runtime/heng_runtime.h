#ifndef HENG_RUNTIME_H
#define HENG_RUNTIME_H
#include <stdint.h>
#include <stddef.h>
typedef struct HObject HObject;
typedef struct HFrame HFrame;
typedef struct HValue { int tag; union { int64_t i; double f; HObject *obj; } as; } HValue;
struct HFrame { HValue *slots; size_t count; HFrame *previous; };
enum { H_VOID, H_I64, H_F64, H_F32, H_BOOL, H_STRING, H_LIST, H_RECORD, H_TENSOR, H_SPARSE, H_RESULT };
HValue hv_void(void);
HValue hv_i64(int64_t value);
HValue hv_f64(double value);
HValue hv_f32(double value);
HValue hv_bool(int value);
HValue hv_string(const char *value);
HValue hv_list(int64_t count, HValue *values);
HValue hv_record(const char *type, int64_t count, const char **names, HValue *values);
HValue hv_field(HValue value, const char *name);
void hv_set_field(HValue value, const char *name, HValue item);
HValue hv_index(HValue value, HValue index);
void hv_set_index(HValue value, HValue index, HValue item);
HValue hv_binary(const char *operation, HValue left, HValue right);
HValue hv_unary(const char *operation, HValue value);
HValue hv_native(const char *name, int64_t count, HValue *arguments);
int hv_truth(HValue value);
int64_t hv_length(HValue value);
int64_t hv_integer(HValue value);
double hv_number(HValue value);
void hv_frame_push(HFrame *frame, HValue *slots, size_t count);
void hv_frame_pop(HFrame *frame);
void hv_safepoint(void);
void hv_site(const char *file, int line, int column);
void hv_fail(const char *message);
void hv_shutdown(void);
#endif
