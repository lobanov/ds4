#include "ds4.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

typedef enum {
    SPEC_MODE_ARGMAX,
    SPEC_MODE_SAMPLE,
    SPEC_MODE_SPECULATIVE_ARGMAX,
    SPEC_MODE_TEACHER_FORCE,
} spec_mode;

typedef struct {
    const char *model_path;
    const char *mtp_path;
    const char *dspark_path;
    const char *bulk_config_path;
    const char *jsonl_path;
    const char *default_system;
    const char *expert_profile_path;
    ds4_backend backend;
    int threads;
    int ctx_alloc;
    int power_percent;
    uint32_t prefill_chunk;
    uint32_t ssd_streaming_cache_experts;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_preload_experts;
    uint64_t simulate_used_memory_bytes;
    bool warm_weights;
    bool quality;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    char *dump_hidden_dir;
    char *force_tokens_dir;
    char *dump_logprobs_jsonl;   /* single appended JSONL of per-anchor top-k logprobs across all prompts */
    int   logprobs_top_k;       /* top-k for the logprobs dump (default 128, max 128) */
    char *rewrite_frontier_path;
} spec_bench_config;

typedef struct {
    char *id;
    char *label;
    char *prompt_path;
    char *chat_prompt_path;
    char *system;
    spec_mode mode;
    int frontier_tokens;
    int gen_tokens;
    float temperature;
    int top_k;
    float top_p;
    float min_p;
    uint64_t seed;
    bool exclude_eos;
    int think_mode;  /* DS4_THINK_* (0=NONE); set via run config "think":"high" */
    int line_no;
} spec_run;

typedef struct {
    spec_run *v;
    int len;
    int cap;
} spec_run_vec;

typedef struct {
    char *prompt_path;
    char *chat_prompt_path;
    char *system;
    ds4_tokens tokens;
} prompt_cache_entry;

typedef struct {
    prompt_cache_entry *v;
    int len;
    int cap;
} prompt_cache;

typedef struct {
    ds4_dspark_cycle_metrics *v;
    int len;
    int cap;
} dspark_cycle_vec;

typedef struct {
    bool ok;
    bool eos_hit;
    int emitted_tokens;
    int cycles;
    int accepted_total;
    int accepted_max;
    bool dspark_metrics_present;
    bool schedule_batched;
    bool scheduled_verify;
    int schedule_batch_limit;
    bool dspark_timing_enabled;
    double prefill_ms;
    double snapshot_ms;
    double decode_ms;
    double restore_ms;
    dspark_cycle_vec dspark_cycles;
    char err[256];
} run_result;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "ds4-spec-bench: out of memory\n");
        exit(1);
    }
    return p;
}

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) {
        fprintf(stderr, "ds4-spec-bench: out of memory\n");
        exit(1);
    }
    return q;
}

static char *xstrdup0(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    char *out = xmalloc(n + 1);
    memcpy(out, s, n + 1);
    return out;
}

static void dspark_cycle_vec_push(dspark_cycle_vec *vec,
                                  const ds4_dspark_cycle_metrics *metric) {
    if (vec->len == vec->cap) {
        int ncap = vec->cap ? vec->cap * 2 : 16;
        vec->v = xrealloc(vec->v, (size_t)ncap * sizeof(vec->v[0]));
        vec->cap = ncap;
    }
    vec->v[vec->len++] = *metric;
}

static double metric_mean_f64(const dspark_cycle_vec *vec, double (*field)(const ds4_dspark_cycle_metrics *)) {
    if (!vec || vec->len == 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < vec->len; i++) sum += field(&vec->v[i]);
    return sum / (double)vec->len;
}

static double metric_mean_i32(const dspark_cycle_vec *vec, int (*field)(const ds4_dspark_cycle_metrics *)) {
    if (!vec || vec->len == 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < vec->len; i++) sum += (double)field(&vec->v[i]);
    return sum / (double)vec->len;
}

static int metric_max_i32(const dspark_cycle_vec *vec, int (*field)(const ds4_dspark_cycle_metrics *)) {
    int best = 0;
    if (!vec || vec->len == 0) return 0;
    best = field(&vec->v[0]);
    for (int i = 1; i < vec->len; i++) {
        int cur = field(&vec->v[i]);
        if (cur > best) best = cur;
    }
    return best;
}

static int metric_field_rows_computed(const ds4_dspark_cycle_metrics *m) { return m->rows_computed; }
static int metric_field_drafted(const ds4_dspark_cycle_metrics *m) { return m->drafted; }
static int metric_field_schedule_batch_limit(const ds4_dspark_cycle_metrics *m) { return m->schedule_batch_limit; }
static int metric_field_verify_n(const ds4_dspark_cycle_metrics *m) { return m->verify_n; }
static int metric_field_verified(const ds4_dspark_cycle_metrics *m) { return m->verified; }
static int metric_field_accepted(const ds4_dspark_cycle_metrics *m) { return m->accepted; }
static double metric_field_decode_ms(const ds4_dspark_cycle_metrics *m) { return m->decode_ms; }
static double metric_field_draft_ms(const ds4_dspark_cycle_metrics *m) { return m->draft_ms; }
static double metric_field_verify_ms(const ds4_dspark_cycle_metrics *m) { return m->verify_ms; }
static double metric_field_total_ms(const ds4_dspark_cycle_metrics *m) { return m->total_ms; }
static double metric_field_push_init_ms(const ds4_dspark_cycle_metrics *m) { return m->push_init_ms; }
static double metric_field_push_verify_ms(const ds4_dspark_cycle_metrics *m) { return m->push_verify_ms; }
static double metric_field_verify_decode_ms(const ds4_dspark_cycle_metrics *m) { return m->verify_decode_ms; }
static double metric_field_logits_read_ms(const ds4_dspark_cycle_metrics *m) { return m->logits_read_ms; }

static int parse_int_arg(const char *s, const char *opt, bool allow_zero) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' ||
        v < (allow_zero ? 0 : 1) || v > INT_MAX) {
        fprintf(stderr, "ds4-spec-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-spec-bench: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static ds4_backend parse_backend(const char *s, const char *opt) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
    if (!strcmp(s, "rocm")) return DS4_BACKEND_CUDA;
#else
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
#endif
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-spec-bench: invalid value for %s: %s\n", opt, s);
    exit(2);
}

static bool parse_gib_arg_local(const char *s, uint64_t *out_bytes) {
    if (!s || !*s || !out_bytes) return false;
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s || !isfinite(v) || v <= 0.0) return false;
    while (*end == ' ') end++;
    if (*end) {
        if (!strcasecmp(end, "g") || !strcasecmp(end, "gb") || !strcasecmp(end, "gib")) {
            /* accepted */
        } else {
            return false;
        }
    }
    double bytes = v * 1024.0 * 1024.0 * 1024.0;
    if (!isfinite(bytes) || bytes <= 0.0 || bytes > (double)UINT64_MAX) return false;
    *out_bytes = (uint64_t)(bytes + 0.5);
    return true;
}

static bool parse_streaming_cache_experts_arg_local(const char *s, uint32_t *out_experts, uint64_t *out_bytes) {
    if (!s || !*s || !out_experts || !out_bytes) return false;
    uint64_t bytes = 0;
    if (parse_gib_arg_local(s, &bytes)) {
        *out_experts = 0;
        *out_bytes = bytes;
        return true;
    }
    char *end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (end == s || *end != '\0' || v == 0 || v > UINT32_MAX) return false;
    *out_experts = (uint32_t)v;
    *out_bytes = 0;
    return true;
}

static void usage(FILE *fp) {
    fprintf(fp,
            "ds4-spec-bench\n\n"
            "Benchmark baseline and speculative decode runs from a JSONL bulk config while\n"
            "reusing one loaded ds4 engine and one session timeline.\n\n"
            "Usage:\n"
            "  ./ds4-spec-bench --bulk-config runs.jsonl [options]\n\n"
            "Engine options:\n"
            "  -m, --model FILE           Target GGUF (default: ds4flash.gguf)\n"
            "      --mtp FILE             Enable MTP drafter\n"
            "      --dspark FILE          Enable DSpark drafter (M3 default speculative stack)\n"
            "      --backend NAME         metal | cuda | rocm | cpu\n"
            "      --metal | --cuda | --rocm | --cpu\n"
            "  -t, --threads N            CPU threads\n"
            "      --ctx-alloc N          Fixed session context; default is max(frontier+gen+1)\n"
            "      --prefill-chunk N      Target prefill chunk\n"
            "      --power N              Engine power percent\n"
            "      --quality              Enable quality mode\n"
            "      --warm-weights         Warm weights on load\n"
            "      --expert-profile FILE  Routed-expert profile path\n"
            "      --ssd-streaming\n"
            "      --ssd-streaming-cold\n"
            "      --ssd-streaming-cache-experts N|MGB\n"
            "      --ssd-streaming-preload-experts N\n"
            "      --simulate-used-memory NGB\n\n"
            "Bulk-run options:\n"
            "      --bulk-config FILE     Required JSONL config file\n"
            "      --jsonl-out FILE       Write run summaries there (default: stdout)\n"
            "      --default-system TEXT  Default chat system prompt\n\n"
            "Bulk config JSONL schema (one JSON object per line):\n"
            "  {\"id\":\"name\",\"mode\":\"speculative_argmax\",\"prompt_file\":\"prompt.txt\",\n"
            "   \"frontier_tokens\":8192,\"gen_tokens\":64}\n"
            "  {\"mode\":\"argmax\",\"chat_prompt_file\":\"chat.txt\",\"system\":\"You are terse.\",\n"
            "   \"frontier_tokens\":4096,\"gen_tokens\":32,\"exclude_eos\":true}\n"
            "  {\"mode\":\"sample\",\"prompt_file\":\"prompt.txt\",\"frontier_tokens\":2048,\n"
            "   \"gen_tokens\":64,\"temperature\":1.0,\"top_k\":50,\"top_p\":0.95,\n"
            "   \"min_p\":0.05,\"seed\":123}\n\n"
            "Modes:\n"
            "  argmax              Plain greedy decode\n"
            "  sample              Plain sampled decode\n"
            "  speculative_argmax  Exact speculative path via ds4_session_eval_speculative_argmax()\n");
}

static spec_bench_config parse_options(int argc, char **argv) {
    spec_bench_config c = {
        .model_path = "ds4flash.gguf",
        .backend = default_backend(),
        .default_system = "You are a helpful assistant.",
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp")) {
            c.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dspark")) {
            c.dspark_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--bulk-config")) {
            c.bulk_config_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--rewrite-frontier")) {
            c.rewrite_frontier_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--jsonl-out")) {
            c.jsonl_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--default-system")) {
            c.default_system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.threads = parse_int_arg(need_arg(&i, argc, argv, arg), arg, false);
        } else if (!strcmp(arg, "--ctx-alloc")) {
            c.ctx_alloc = parse_int_arg(need_arg(&i, argc, argv, arg), arg, false);
        } else if (!strcmp(arg, "--prefill-chunk")) {
            c.prefill_chunk = (uint32_t)parse_int_arg(need_arg(&i, argc, argv, arg), arg, false);
        } else if (!strcmp(arg, "--power")) {
            c.power_percent = parse_int_arg(need_arg(&i, argc, argv, arg), arg, false);
        } else if (!strcmp(arg, "--expert-profile")) {
            c.expert_profile_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--metal")) {
            c.backend = DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
        } else if (!strcmp(arg, "--rocm")) {
            c.backend = DS4_BACKEND_CUDA;
#else
        } else if (!strcmp(arg, "--cuda")) {
            c.backend = DS4_BACKEND_CUDA;
#endif
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--quality")) {
            c.quality = true;
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else if (!strcmp(arg, "--ssd-streaming")) {
            c.ssd_streaming = true;
        } else if (!strcmp(arg, "--ssd-streaming-cold")) {
            c.ssd_streaming_cold = true;
        } else if (!strcmp(arg, "--dump-hidden-dir")) {
            c.dump_hidden_dir = xstrdup0(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--force-tokens-dir")) {
            c.force_tokens_dir = xstrdup0(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--dump-logprobs-jsonl")) {
            c.dump_logprobs_jsonl = xstrdup0(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--logprobs-top-k")) {
            c.logprobs_top_k = atoi(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--ssd-streaming-cache-experts")) {
            uint32_t experts = 0;
            uint64_t bytes = 0;
            if (!parse_streaming_cache_experts_arg_local(need_arg(&i, argc, argv, arg), &experts, &bytes)) {
                fprintf(stderr,
                        "ds4-spec-bench: --ssd-streaming-cache-experts must be a positive count or <number>GB\n");
                exit(2);
            }
            c.ssd_streaming_cache_experts = experts;
            c.ssd_streaming_cache_bytes = bytes;
        } else if (!strcmp(arg, "--ssd-streaming-preload-experts")) {
            c.ssd_streaming_preload_experts = (uint32_t)parse_int_arg(need_arg(&i, argc, argv, arg), arg, false);
        } else if (!strcmp(arg, "--simulate-used-memory")) {
            if (!parse_gib_arg_local(need_arg(&i, argc, argv, arg), &c.simulate_used_memory_bytes)) {
                fprintf(stderr, "ds4-spec-bench: --simulate-used-memory must be a positive GiB value\n");
                exit(2);
            }
        } else {
            fprintf(stderr, "ds4-spec-bench: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }

    if (!c.bulk_config_path) {
        fprintf(stderr, "ds4-spec-bench: --bulk-config is required\n");
        exit(2);
    }
    if (c.mtp_path && c.dspark_path) {
        fprintf(stderr, "ds4-spec-bench: choose either --mtp or --dspark, not both\n");
        exit(2);
    }
    return c;
}

#define JSON_MAX_NESTING 64

static void json_ws(const char **p) {
    while (**p == ' ' || **p == '\t' || **p == '\r' || **p == '\n') (*p)++;
}

static bool json_lit(const char **p, const char *lit) {
    size_t n = strlen(lit);
    if (strncmp(*p, lit, n) != 0) return false;
    *p += n;
    return true;
}

static int json_hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

static bool json_u16(const char **p, uint32_t *out) {
    if ((*p)[0] != '\\' || (*p)[1] != 'u') return false;
    uint32_t cp = 0;
    for (int i = 0; i < 4; i++) {
        int h = json_hex((*p)[2 + i]);
        if (h < 0) return false;
        cp = (cp << 4) | (uint32_t)h;
    }
    *p += 6;
    *out = cp;
    return true;
}

static void utf8_append(char **buf, size_t *len, size_t *cap, uint32_t cp) {
    unsigned char tmp[4];
    size_t n = 0;
    if (cp <= 0x7f) {
        tmp[n++] = (unsigned char)cp;
    } else if (cp <= 0x7ff) {
        tmp[n++] = 0xc0u | (unsigned char)(cp >> 6);
        tmp[n++] = 0x80u | (unsigned char)(cp & 0x3f);
    } else if (cp <= 0xffff) {
        tmp[n++] = 0xe0u | (unsigned char)(cp >> 12);
        tmp[n++] = 0x80u | (unsigned char)((cp >> 6) & 0x3f);
        tmp[n++] = 0x80u | (unsigned char)(cp & 0x3f);
    } else {
        tmp[n++] = 0xf0u | (unsigned char)(cp >> 18);
        tmp[n++] = 0x80u | (unsigned char)((cp >> 12) & 0x3f);
        tmp[n++] = 0x80u | (unsigned char)((cp >> 6) & 0x3f);
        tmp[n++] = 0x80u | (unsigned char)(cp & 0x3f);
    }
    if (*len + n + 1 > *cap) {
        *cap = (*cap ? *cap * 2 : 64);
        while (*len + n + 1 > *cap) *cap *= 2;
        *buf = xrealloc(*buf, *cap);
    }
    memcpy(*buf + *len, tmp, n);
    *len += n;
    (*buf)[*len] = '\0';
}

static bool json_string(const char **p, char **out) {
    json_ws(p);
    if (**p != '"') return false;
    (*p)++;
    char *buf = NULL;
    size_t len = 0;
    size_t cap = 0;
    while (**p && **p != '"') {
        if ((unsigned char)**p < 0x20) goto fail;
        if (**p != '\\') {
            if (len + 2 > cap) {
                cap = cap ? cap * 2 : 64;
                if (len + 2 > cap) cap = len + 2;
                buf = xrealloc(buf, cap);
            }
            buf[len++] = *(*p)++;
            buf[len] = '\0';
            continue;
        }
        (*p)++;
        switch (**p) {
        case '"': case '\\': case '/':
            utf8_append(&buf, &len, &cap, (unsigned char)**p);
            (*p)++;
            break;
        case 'b': utf8_append(&buf, &len, &cap, '\b'); (*p)++; break;
        case 'f': utf8_append(&buf, &len, &cap, '\f'); (*p)++; break;
        case 'n': utf8_append(&buf, &len, &cap, '\n'); (*p)++; break;
        case 'r': utf8_append(&buf, &len, &cap, '\r'); (*p)++; break;
        case 't': utf8_append(&buf, &len, &cap, '\t'); (*p)++; break;
        case 'u': {
            (*p)--;
            uint32_t cp = 0;
            if (!json_u16(p, &cp)) goto fail;
            if (cp >= 0xd800 && cp <= 0xdbff) {
                const char *save = *p;
                uint32_t lo = 0;
                if (json_u16(p, &lo) && lo >= 0xdc00 && lo <= 0xdfff) {
                    cp = 0x10000u + (((cp - 0xd800u) << 10) | (lo - 0xdc00u));
                } else {
                    *p = save;
                }
            }
            utf8_append(&buf, &len, &cap, cp);
            break;
        }
        default:
            goto fail;
        }
    }
    if (**p != '"') goto fail;
    (*p)++;
    if (!buf) buf = xstrdup0("");
    *out = buf;
    return true;
fail:
    free(buf);
    return false;
}

static bool json_number(const char **p, double *out) {
    json_ws(p);
    char *end = NULL;
    double v = strtod(*p, &end);
    if (end == *p || !isfinite(v)) return false;
    *p = end;
    *out = v;
    return true;
}

static bool json_int(const char **p, int *out) {
    double v = 0.0;
    if (!json_number(p, &v)) return false;
    if (v < (double)INT_MIN || v > (double)INT_MAX || floor(v) != v) return false;
    *out = (int)v;
    return true;
}

static bool json_bool(const char **p, bool *out) {
    json_ws(p);
    if (json_lit(p, "true")) {
        *out = true;
        return true;
    }
    if (json_lit(p, "false")) {
        *out = false;
        return true;
    }
    return false;
}

static bool json_skip_value_depth(const char **p, int depth);

static bool json_skip_array_depth(const char **p, int depth) {
    if (depth > JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == ',') {
            (*p)++;
            json_ws(p);
        } else if (**p != ']') {
            return false;
        }
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}

static bool json_skip_object_depth(const char **p, int depth) {
    if (depth > JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        free(key);
        json_ws(p);
        if (**p != ':') return false;
        (*p)++;
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == ',') {
            (*p)++;
            json_ws(p);
        } else if (**p != '}') {
            return false;
        }
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}

static bool json_skip_value_depth(const char **p, int depth) {
    json_ws(p);
    if (**p == '"') {
        char *s = NULL;
        bool ok = json_string(p, &s);
        free(s);
        return ok;
    }
    if (**p == '{') return json_skip_object_depth(p, depth);
    if (**p == '[') return json_skip_array_depth(p, depth);
    if (json_lit(p, "true") || json_lit(p, "false") || json_lit(p, "null")) return true;
    double v = 0.0;
    return json_number(p, &v);
}

static bool json_skip_value(const char **p) {
    return json_skip_value_depth(p, 0);
}

static const char *mode_name(spec_mode mode) {
    switch (mode) {
    case SPEC_MODE_ARGMAX: return "argmax";
    case SPEC_MODE_SAMPLE: return "sample";
    case SPEC_MODE_SPECULATIVE_ARGMAX: return "speculative_argmax";
    case SPEC_MODE_TEACHER_FORCE: return "teacher_force";
    }
    return "unknown";
}

static bool parse_mode(const char *s, spec_mode *out) {
    if (!strcmp(s, "argmax") || !strcmp(s, "greedy")) {
        *out = SPEC_MODE_ARGMAX;
        return true;
    }
    if (!strcmp(s, "sample")) {
        *out = SPEC_MODE_SAMPLE;
        return true;
    }
    if (!strcmp(s, "speculative") || !strcmp(s, "speculative_argmax")) {
        *out = SPEC_MODE_SPECULATIVE_ARGMAX;
        return true;
    }
    if (!strcmp(s, "teacher_force") || !strcmp(s, "teacher-force") || !strcmp(s, "force")) {
        *out = SPEC_MODE_TEACHER_FORCE;
        return true;
    }
    return false;
}

static void run_vec_push(spec_run_vec *v, spec_run run) {
    if (v->len == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 32;
        v->v = xrealloc(v->v, (size_t)v->cap * sizeof(v->v[0]));
    }
    v->v[v->len++] = run;
}

static void run_free(spec_run *r) {
    if (!r) return;
    free(r->id);
    free(r->label);
    free(r->prompt_path);
    free(r->chat_prompt_path);
    free(r->system);
    memset(r, 0, sizeof(*r));
}

static char *read_text_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long n = ftell(fp);
    if (n < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }
    char *buf = xmalloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    buf[n] = '\0';
    return buf;
}

static bool parse_run_json(const char *line, int line_no, const char *default_system, spec_run *out, char *err, size_t errlen) {
    spec_run run = {
        .mode = SPEC_MODE_SPECULATIVE_ARGMAX,
        .temperature = 0.0f,
        .top_k = 0,
        .top_p = DS4_DEFAULT_TOP_P,
        .min_p = DS4_DEFAULT_MIN_P,
        .seed = (uint64_t)line_no,
        .exclude_eos = true,
        .line_no = line_no,
    };
    if (default_system) run.system = xstrdup0(default_system);

    const char *p = line;
    json_ws(&p);
    if (*p != '{') {
        snprintf(err, errlen, "line %d: expected JSON object", line_no);
        run_free(&run);
        return false;
    }
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) {
            snprintf(err, errlen, "line %d: invalid object key", line_no);
            run_free(&run);
            return false;
        }
        json_ws(&p);
        if (*p != ':') {
            snprintf(err, errlen, "line %d: missing ':' after key", line_no);
            free(key);
            run_free(&run);
            return false;
        }
        p++;
        if (!strcmp(key, "id")) {
            free(run.id);
            if (!json_string(&p, &run.id)) goto bad_value;
        } else if (!strcmp(key, "label")) {
            free(run.label);
            if (!json_string(&p, &run.label)) goto bad_value;
        } else if (!strcmp(key, "prompt_file")) {
            free(run.prompt_path);
            if (!json_string(&p, &run.prompt_path)) goto bad_value;
        } else if (!strcmp(key, "chat_prompt_file")) {
            free(run.chat_prompt_path);
            if (!json_string(&p, &run.chat_prompt_path)) goto bad_value;
        } else if (!strcmp(key, "system")) {
            free(run.system);
            run.system = NULL;
            if (json_lit(&p, "null")) {
                /* cleared */
            } else if (!json_string(&p, &run.system)) {
                goto bad_value;
            }
        } else if (!strcmp(key, "mode")) {
            char *mode_s = NULL;
            if (!json_string(&p, &mode_s)) goto bad_value;
            if (!parse_mode(mode_s, &run.mode)) {
                snprintf(err, errlen, "line %d: unknown mode %s", line_no, mode_s);
                free(mode_s);
                free(key);
                run_free(&run);
                return false;
            }
            free(mode_s);
        } else if (!strcmp(key, "frontier_tokens") || !strcmp(key, "ctx_tokens") || !strcmp(key, "frontier")) {
            if (!json_int(&p, &run.frontier_tokens)) goto bad_value;
        } else if (!strcmp(key, "gen_tokens") || !strcmp(key, "tokens")) {
            if (!json_int(&p, &run.gen_tokens)) goto bad_value;
        } else if (!strcmp(key, "temperature")) {
            double v = 0.0;
            if (!json_number(&p, &v)) goto bad_value;
            run.temperature = (float)v;
        } else if (!strcmp(key, "top_k")) {
            if (!json_int(&p, &run.top_k)) goto bad_value;
        } else if (!strcmp(key, "top_p")) {
            double v = 0.0;
            if (!json_number(&p, &v)) goto bad_value;
            run.top_p = (float)v;
        } else if (!strcmp(key, "min_p")) {
            double v = 0.0;
            if (!json_number(&p, &v)) goto bad_value;
            run.min_p = (float)v;
        } else if (!strcmp(key, "seed")) {
            double v = 0.0;
            if (!json_number(&p, &v) || v < 0.0 || v > (double)UINT64_MAX || floor(v) != v) goto bad_value;
            run.seed = (uint64_t)v;
        } else if (!strcmp(key, "exclude_eos")) {
            if (!json_bool(&p, &run.exclude_eos)) goto bad_value;
        } else if (!strcmp(key, "think")) {
            char *tv = NULL;
            if (!json_string(&p, &tv)) goto bad_value;
            if (tv && !strcmp(tv, "high")) run.think_mode = DS4_THINK_HIGH;
            else if (tv && !strcmp(tv, "max")) run.think_mode = DS4_THINK_MAX;
            else run.think_mode = DS4_THINK_NONE;
            free(tv);
        } else {
            if (!json_skip_value(&p)) goto bad_value;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') {
            p++;
            json_ws(&p);
        } else if (*p != '}') {
            snprintf(err, errlen, "line %d: expected ',' or '}'", line_no);
            run_free(&run);
            return false;
        }
        continue;
bad_value:
        snprintf(err, errlen, "line %d: invalid value for key %s", line_no, key ? key : "(unknown)");
        free(key);
        run_free(&run);
        return false;
    }
    if (*p != '}') {
        snprintf(err, errlen, "line %d: unterminated JSON object", line_no);
        run_free(&run);
        return false;
    }
    p++;
    json_ws(&p);
    if (*p) {
        snprintf(err, errlen, "line %d: trailing content after JSON object", line_no);
        run_free(&run);
        return false;
    }

    if (!!run.prompt_path == !!run.chat_prompt_path) {
        snprintf(err, errlen, "line %d: specify exactly one of prompt_file or chat_prompt_file", line_no);
        run_free(&run);
        return false;
    }
    if (run.frontier_tokens <= 0) {
        snprintf(err, errlen, "line %d: frontier_tokens must be positive", line_no);
        run_free(&run);
        return false;
    }
    if (run.gen_tokens < 0) {
        snprintf(err, errlen, "line %d: gen_tokens must be non-negative", line_no);
        run_free(&run);
        return false;
    }
    if (run.temperature < 0.0f) {
        snprintf(err, errlen, "line %d: temperature must be >= 0", line_no);
        run_free(&run);
        return false;
    }
    if (run.top_p <= 0.0f || run.top_p > 1.0f) {
        snprintf(err, errlen, "line %d: top_p must be in (0, 1]", line_no);
        run_free(&run);
        return false;
    }
    if (run.min_p < 0.0f || run.min_p > 1.0f) {
        snprintf(err, errlen, "line %d: min_p must be in [0, 1]", line_no);
        run_free(&run);
        return false;
    }
    if (run.mode == SPEC_MODE_SPECULATIVE_ARGMAX && run.temperature > 0.0f) {
        snprintf(err, errlen, "line %d: speculative_argmax only supports temperature=0", line_no);
        run_free(&run);
        return false;
    }
    *out = run;
    return true;
}

static char *read_line(FILE *fp) {
    size_t cap = 0;
    size_t len = 0;
    char *buf = NULL;
    for (;;) {
        int ch = fgetc(fp);
        if (ch == EOF) break;
        if (len + 2 > cap) {
            cap = cap ? cap * 2 : 256;
            buf = xrealloc(buf, cap);
        }
        buf[len++] = (char)ch;
        if (ch == '\n') break;
    }
    if (!buf && feof(fp)) return NULL;
    if (!buf) buf = xstrdup0("");
    buf[len] = '\0';
    return buf;
}

static void trim_line(char *s) {
    if (!s) return;
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r' || s[n - 1] == ' ' || s[n - 1] == '\t')) {
        s[--n] = '\0';
    }
    char *p = s;
    while (*p == ' ' || *p == '\t') p++;
    if (p != s) memmove(s, p, strlen(p) + 1);
}

static spec_run_vec load_runs(const spec_bench_config *cfg) {
    FILE *fp = fopen(cfg->bulk_config_path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4-spec-bench: failed to open %s: %s\n", cfg->bulk_config_path, strerror(errno));
        exit(1);
    }
    spec_run_vec runs = {0};
    int line_no = 0;
    for (;;) {
        char *line = read_line(fp);
        if (!line) break;
        line_no++;
        trim_line(line);
        if (!line[0]) {
            free(line);
            continue;
        }
        spec_run run = {0};
        char err[256];
        if (!parse_run_json(line, line_no, cfg->default_system, &run, err, sizeof(err))) {
            fprintf(stderr, "ds4-spec-bench: %s\n", err);
            free(line);
            fclose(fp);
            for (int i = 0; i < runs.len; i++) run_free(&runs.v[i]);
            free(runs.v);
            exit(2);
        }
        run_vec_push(&runs, run);
        free(line);
    }
    fclose(fp);
    if (runs.len == 0) {
        fprintf(stderr, "ds4-spec-bench: no runs found in %s\n", cfg->bulk_config_path);
        exit(2);
    }
    return runs;
}

static void prompt_cache_push(prompt_cache *cache, prompt_cache_entry entry) {
    if (cache->len == cache->cap) {
        cache->cap = cache->cap ? cache->cap * 2 : 16;
        cache->v = xrealloc(cache->v, (size_t)cache->cap * sizeof(cache->v[0]));
    }
    cache->v[cache->len++] = entry;
}

static bool prompt_cache_matches(const prompt_cache_entry *e, const spec_run *run) {
    if (!!e->prompt_path != !!run->prompt_path) return false;
    if (!!e->chat_prompt_path != !!run->chat_prompt_path) return false;
    if (!!e->system != !!run->system) return false;
    if (e->prompt_path && strcmp(e->prompt_path, run->prompt_path) != 0) return false;
    if (e->chat_prompt_path && strcmp(e->chat_prompt_path, run->chat_prompt_path) != 0) return false;
    if (e->system && strcmp(e->system, run->system) != 0) return false;
    return true;
}

static const ds4_tokens *prompt_cache_get(
        prompt_cache *cache,
        ds4_engine   *engine,
        const spec_run *run,
        char        *err,
        size_t       errlen) {
    for (int i = 0; i < cache->len; i++) {
        if (prompt_cache_matches(&cache->v[i], run)) return &cache->v[i].tokens;
    }
    char *text = read_text_file(run->prompt_path ? run->prompt_path : run->chat_prompt_path);
    if (!text) {
        snprintf(err, errlen, "failed to read %s: %s",
                 run->prompt_path ? run->prompt_path : run->chat_prompt_path,
                 strerror(errno));
        return NULL;
    }
    prompt_cache_entry entry = {
        .prompt_path = xstrdup0(run->prompt_path),
        .chat_prompt_path = xstrdup0(run->chat_prompt_path),
        .system = xstrdup0(run->system),
    };
    if (run->chat_prompt_path) {
        ds4_think_mode tm = (run->think_mode > 0) ? (ds4_think_mode)run->think_mode : DS4_THINK_NONE;
        ds4_encode_chat_prompt(engine, run->system, text, tm, &entry.tokens);
    } else {
        ds4_tokenize_text(engine, text, &entry.tokens);
    }
    free(text);
    prompt_cache_push(cache, entry);
    return &cache->v[cache->len - 1].tokens;
}

static void json_write_string(FILE *fp, const char *s) {
    fputc('"', fp);
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
            switch (*p) {
            case '"': fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\b': fputs("\\b", fp); break;
            case '\f': fputs("\\f", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            default:
                if (*p < 0x20) fprintf(fp, "\\u%04x", (unsigned)*p);
                else fputc((char)*p, fp);
                break;
            }
        }
    }
    fputc('"', fp);
}

static int auto_ctx_alloc(const spec_run_vec *runs) {
    int max_need = 0;
    for (int i = 0; i < runs->len; i++) {
        const spec_run *r = &runs->v[i];
        if (r->frontier_tokens > INT_MAX - r->gen_tokens - 1) {
            fprintf(stderr, "ds4-spec-bench: run line %d requires too much context\n", r->line_no);
            exit(2);
        }
        int need = r->frontier_tokens + r->gen_tokens + 1;
        if (need > max_need) max_need = need;
    }
    return max_need > 0 ? max_need : 1;
}

/* m3: eager frontier>prompt validation — tokenize each run's prompt at startup
 * + report any entry whose frontier_tokens exceeds the prompt's token count. The
 * bench REFUSES to run if any entry is invalid (previously it silently dropped
 * them, producing a biased subset + missing prompts). Use --rewrite-frontier OUT
 * to emit a corrected config (frontier = prompt_tokens). */
static int validate_frontier_runs(spec_run_vec *runs, prompt_cache *cache,
                                  ds4_engine *engine) {
    int invalid = 0;
    for (int i = 0; i < runs->len; i++) {
        spec_run *run = &runs->v[i];
        char err[256] = {0};
        const ds4_tokens *tokens = prompt_cache_get(cache, engine, run, err, sizeof(err));
        if (!tokens) {
            fprintf(stderr, "ds4-spec-bench: INVALID line %d (%s): tokenize failed: %s\n",
                    run->line_no, run->id ? run->id : "?", err);
            invalid++;
        } else if ((int)tokens->len < run->frontier_tokens) {
            fprintf(stderr, "ds4-spec-bench: INVALID line %d (%s): frontier %d > prompt %d tokens (set frontier_tokens<=%d)\n",
                    run->line_no, run->id ? run->id : "?", run->frontier_tokens,
                    (int)tokens->len, (int)tokens->len);
            invalid++;
        }
    }
    return invalid;
}

/* m3: --rewrite-frontier OUT — tokenize each run's prompt + emit a corrected
 * config with frontier_tokens = the prompt's token count (so no entry is invalid).
 * The bench REFUSES to run an invalid config (no silent skips); this regenerates
 * a valid one to re-run with. */
static int rewrite_frontier_config(spec_run_vec *runs, prompt_cache *cache,
                                   ds4_engine *engine, const char *out_path) {
    FILE *fp = fopen(out_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4-spec-bench: failed to open %s for rewrite: %s\n", out_path, strerror(errno));
        return 1;
    }
    for (int i = 0; i < runs->len; i++) {
        spec_run *run = &runs->v[i];
        char err[256] = {0};
        const ds4_tokens *tokens = prompt_cache_get(cache, engine, run, err, sizeof(err));
        int prompt_len = tokens ? (int)tokens->len : run->frontier_tokens;
        fprintf(fp, "{\"frontier_tokens\":%d", prompt_len);  /* corrected; first field */
        fprintf(fp, ",\"mode\":\"%s\"", mode_name(run->mode));
        fprintf(fp, ",\"gen_tokens\":%d", run->gen_tokens);
        if (run->id) { fprintf(fp, ",\"id\":"); json_write_string(fp, run->id); }
        if (run->prompt_path) { fprintf(fp, ",\"prompt_file\":"); json_write_string(fp, run->prompt_path); }
        if (run->chat_prompt_path) { fprintf(fp, ",\"chat_prompt_file\":"); json_write_string(fp, run->chat_prompt_path); }
        if (run->system) { fprintf(fp, ",\"system\":"); json_write_string(fp, run->system); }
        if (run->temperature != 0.0f) fprintf(fp, ",\"temperature\":%g", (double)run->temperature);
        if (run->top_k > 0) fprintf(fp, ",\"top_k\":%d", run->top_k);
        if (run->top_p != 1.0f) fprintf(fp, ",\"top_p\":%g", (double)run->top_p);
        if (run->min_p > 0.0f) fprintf(fp, ",\"min_p\":%g", (double)run->min_p);
        if (run->seed != 0) fprintf(fp, ",\"seed\":%llu", (unsigned long long)run->seed);
        if (run->exclude_eos) fprintf(fp, ",\"exclude_eos\":true");
        if (run->think_mode != 0) fprintf(fp, ",\"think\":%d", run->think_mode);
        fprintf(fp, "}\n");
    }
    fclose(fp);
    fprintf(stderr, "ds4-spec-bench: rewrote %d run(s) -> %s (frontier_tokens = prompt token count)\n", runs->len, out_path);
    return 0;
}

static const char *active_drafter_name(ds4_engine *engine) {
    if (ds4_engine_has_dspark(engine)) return "dspark";
    if (ds4_engine_has_mtp(engine)) return "mtp";
    return "none";
}

/* Read whitespace-separated forced token IDs from `path` into `out` (up to `max`).
 * Returns the count read, or -1 if the file cannot be opened. Used by the
 * teacher_force mode to drive the model through a provided (e.g. FP) trajectory
 * so DS4_DSPARK_DUMP_HIDDEN captures H on that common trajectory. */
static int read_force_tokens(const char *path, int *out, int max) {
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    int n = 0, tok, got;
    while (n < max && (got = fscanf(fp, "%d", &tok)) == 1) out[n++] = tok;
    fclose(fp);
    return n;
}

/* Single-append-JSONL top-k logprobs dump. Fires once per argmax step (the logits predict
 * the step's token; scores[0].id == the greedy). All prompts' records go to ONE file (the path
 * from DS4_BENCH_DUMP_LOGPROBS_JSONL), so the capture is a single JSONL, not many small files.
 * Record: {"id":<prompt_id>,"pos":<step>,"sel":<greedy_tok>,"top":[[id,logprob],...]} */
static void dump_logprobs_jsonl_if_enabled(ds4_session *s, const char *prompt_id, int pos, int selected) {
    const char *path = getenv("DS4_BENCH_DUMP_LOGPROBS_JSONL");
    if (!path || !path[0] || !s) return;
    int k = 128;
    const char *ke = getenv("DS4_BENCH_LOGPROBS_TOP_K");
    if (ke && atoi(ke) > 0 && atoi(ke) <= 128) k = atoi(ke);
    static FILE *fp = NULL;
    static char open_path[1024] = {0};
    if (!fp || strcmp(open_path, path) != 0) {
        if (fp) fclose(fp);
        fp = fopen(path, "ab");  /* append: one file across all prompts */
        if (!fp) return;
        strncpy(open_path, path, sizeof(open_path) - 1);
    }
    ds4_token_score scores[128];
    int n = ds4_session_top_logprobs(s, scores, k);
    fprintf(fp, "{\"id\":");
    json_write_string(fp, prompt_id ? prompt_id : "");
    fprintf(fp, ",\"pos\":%d,\"sel\":%d,\"top\":[", pos, selected);
    for (int i = 0; i < n && i < k && scores[i].id >= 0; i++) {
        if (i) fputc(',', fp);
        fprintf(fp, "[%d,%.9g]", scores[i].id, scores[i].logprob);
    }
    fputs("]}\n", fp);
    fflush(fp);
}

static run_result execute_run(
        ds4_engine        *engine,
        int                ctx_alloc,
        const spec_run    *run,
        const ds4_tokens  *tokens) {
    run_result res = {.ok = false};
    char err[256] = {0};
    ds4_session *session = NULL;
    if (tokens->len < run->frontier_tokens) {
        snprintf(res.err, sizeof(res.err),
                 "prompt has %d tokens, need frontier %d",
                 tokens->len, run->frontier_tokens);
        return res;
    }
    if (ds4_session_create(&session, engine, ctx_alloc) != 0 || !session) {
        snprintf(res.err, sizeof(res.err), "failed to create session");
        return res;
    }

    ds4_tokens prefix = {
        .v = tokens->v,
        .len = run->frontier_tokens,
        .cap = run->frontier_tokens,
    };

    const double prefill_t0 = now_sec();
    if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
        snprintf(res.err, sizeof(res.err), "sync failed: %s", err);
        return res;
    }
    const double prefill_t1 = now_sec();
    res.prefill_ms = (prefill_t1 - prefill_t0) * 1000.0;

    const int eos = ds4_token_eos(engine);
    const double decode_t0 = now_sec();
    uint64_t rng = run->seed;
    int emitted = 0;
    int cycles = 0;
    int accepted_total = 0;
    int accepted_max = 0;
    int tok_out[8192];
    int tok_n = 0;
    int force_tok[8192];   /* teacher_force: the forced (FP) trajectory tokens */
    int force_n = 0;
    int argmax_out[8192];  /* teacher_force: the model argmax (Y_iq2_tf) per step */
    int argmax_n = 0;
    if (run->mode == SPEC_MODE_TEACHER_FORCE) {
        const char *fpath = getenv("DS4_FORCE_TOKENS_PATH");
        if (!fpath || !fpath[0]) {
            snprintf(res.err, sizeof(res.err), "teacher_force: DS4_FORCE_TOKENS_PATH not set (use --force-tokens-dir)");
            goto done;
        }
        force_n = read_force_tokens(fpath, force_tok, 8192);
        if (force_n <= 0) {
            snprintf(res.err, sizeof(res.err), "teacher_force: no forced tokens read from %s", fpath);
            goto done;
        }
    }

    while (emitted < run->gen_tokens) {
        const int remaining = run->gen_tokens - emitted;
        int produced = 0;
        if (run->mode == SPEC_MODE_ARGMAX) {
            int token = run->exclude_eos ? ds4_session_argmax_excluding(session, eos)
                                         : ds4_session_argmax(session);
            if (token < 0) {
                snprintf(res.err, sizeof(res.err), "failed to select greedy token");
                goto done;
            }
            if (!run->exclude_eos && token == eos) res.eos_hit = true;
            dump_logprobs_jsonl_if_enabled(session, run->id, emitted, token);
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                snprintf(res.err, sizeof(res.err), "decode failed: %s", err);
                goto done;
            }
            if (tok_n < 8192) tok_out[tok_n++] = token;
            produced = 1;
        } else if (run->mode == SPEC_MODE_SAMPLE) {
            int token = ds4_session_sample(session,
                                           run->temperature,
                                           run->top_k,
                                           run->top_p,
                                           run->min_p,
                                           &rng);
            if (token < 0) {
                snprintf(res.err, sizeof(res.err), "failed to sample token");
                goto done;
            }
            if (token == eos) res.eos_hit = true;
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                snprintf(res.err, sizeof(res.err), "decode failed: %s", err);
                goto done;
            }
            if (tok_n < 8192) tok_out[tok_n++] = token;
            produced = 1;
        } else if (run->mode == SPEC_MODE_TEACHER_FORCE) {
            /* Teacher-force: drive the model through the FP greedy tokens. At step k,
             * record the model's logit-argmax (Y_iq2_tf[k] — NOT the forced token) and
             * commit the forced token. DS4_DSPARK_DUMP_HIDDEN captures H_iq2_tf[k].
             * The dumped `tok` is the committed (forced) token = Y_fp[k] (a cross-check). */
            const int k = emitted;
            if (k >= force_n) break;  /* forced tokens exhausted -> normal end */
            int argmax_tok = ds4_session_argmax(session);
            if (argmax_tok < 0) {
                snprintf(res.err, sizeof(res.err), "teacher_force: argmax failed at step %d", k);
                goto done;
            }
            if (argmax_n < 8192) argmax_out[argmax_n++] = argmax_tok;
            const int forced = force_tok[k];
            if (ds4_session_eval(session, forced, err, sizeof(err)) != 0) {
                snprintf(res.err, sizeof(res.err), "teacher_force decode failed: %s", err);
                goto done;
            }
            if (tok_n < 8192) tok_out[tok_n++] = forced;
            produced = 1;
        } else {
            int first = run->exclude_eos ? ds4_session_argmax_excluding(session, eos)
                                         : ds4_session_argmax(session);
            if (first < 0) {
                snprintf(res.err, sizeof(res.err), "failed to select speculative first token");
                goto done;
            }
            int accepted[256];
            int cap = remaining;
            if (cap > (int)(sizeof(accepted) / sizeof(accepted[0]))) cap = (int)(sizeof(accepted) / sizeof(accepted[0]));
            produced = ds4_session_eval_speculative_argmax(session,
                                                           first,
                                                           remaining,
                                                           eos,
                                                           accepted,
                                                           cap,
                                                           err,
                                                           sizeof(err));
            if (produced < 0) {
                snprintf(res.err, sizeof(res.err), "speculative decode failed: %s", err);
                goto done;
            }
            if (ds4_engine_has_dspark(engine)) {
                ds4_dspark_cycle_metrics metric = {0};
                res.dspark_timing_enabled = getenv("DS4_DSPARK_TIMING") != NULL;
                if (ds4_session_get_dspark_last_cycle_metrics(session, &metric) == 0) {
                    res.dspark_metrics_present = true;
                    res.schedule_batched = res.schedule_batched || metric.batched_schedule;
                    res.scheduled_verify = res.scheduled_verify || metric.scheduled_verify;
                    if (metric.schedule_batch_limit > res.schedule_batch_limit) {
                        res.schedule_batch_limit = metric.schedule_batch_limit;
                    }
                    dspark_cycle_vec_push(&res.dspark_cycles, &metric);
                }
            }
            for (int i = 0; i < produced; i++) {
                if (accepted[i] == eos) {
                    res.eos_hit = true;
                    break;
                }
            }
            for (int i = 0; i < produced && tok_n < 8192; i++) tok_out[tok_n++] = accepted[i];
        }

        if (produced <= 0) break;
        emitted += produced;
        cycles++;
        accepted_total += produced;
        if (produced > accepted_max) accepted_max = produced;
        if (res.eos_hit) break;
    }

    res.decode_ms = (now_sec() - decode_t0) * 1000.0;
    res.emitted_tokens = emitted;
    res.cycles = cycles;
    res.accepted_total = accepted_total;
    res.accepted_max = accepted_max;
    {
        const char *dtd = getenv("DS4_BENCH_DUMP_TOKENS");
        if (dtd && run->id && run->id[0]) {
            char path[512];
            snprintf(path, sizeof(path), "%s/%s.ids", dtd, run->id);
            FILE *fp = fopen(path, "w");
            if (fp) { for (int i = 0; i < tok_n; i++) fprintf(fp, "%d\n", tok_out[i]); fclose(fp); }
            snprintf(path, sizeof(path), "%s/%s.txt", dtd, run->id);
            fp = fopen(path, "w");
            if (fp) {
                for (int i = 0; i < tok_n; i++) {
                    size_t plen = 0;
                    char *piece = ds4_token_text(engine, tok_out[i], &plen);
                    if (piece) { fwrite(piece, 1, plen, fp); free(piece); }
                }
                fclose(fp);
            }
        }
    }
    if (run->mode == SPEC_MODE_TEACHER_FORCE && argmax_n > 0) {
        const char *ap = getenv("DS4_IQ2_ARGMAX_PATH");
        if (ap && ap[0]) {
            FILE *af = fopen(ap, "w");
            if (af) {
                for (int i = 0; i < argmax_n; i++)
                    fprintf(af, "%d%s", argmax_out[i], i + 1 < argmax_n ? " " : "");
                fclose(af);
            }
        }
    }
    res.ok = true;

done:
    ds4_session_free(session);
    return res;
}

static void run_result_free(run_result *res) {
    if (!res) return;
    free(res->dspark_cycles.v);
    res->dspark_cycles.v = NULL;
    res->dspark_cycles.len = 0;
    res->dspark_cycles.cap = 0;
}

static void write_result_jsonl(
        FILE                *out,
        const spec_bench_config *cfg,
        ds4_engine          *engine,
        const spec_run      *run,
        const ds4_tokens    *tokens,
        const run_result    *res) {
    fprintf(out, "{");
    fprintf(out, "\"line\":%d", run->line_no);
    fprintf(out, ",\"id\":");
    json_write_string(out, run->id);
    fprintf(out, ",\"label\":");
    json_write_string(out, run->label);
    fprintf(out, ",\"mode\":");
    json_write_string(out, mode_name(run->mode));
    fprintf(out, ",\"status\":");
    json_write_string(out, res->ok ? "ok" : "error");
    fprintf(out, ",\"model\":");
    json_write_string(out, cfg->model_path);
    fprintf(out, ",\"backend\":");
    json_write_string(out, ds4_backend_name(cfg->backend));
    fprintf(out, ",\"drafter\":");
    json_write_string(out, active_drafter_name(engine));
    fprintf(out, ",\"has_mtp\":%s", ds4_engine_has_mtp(engine) ? "true" : "false");
    fprintf(out, ",\"has_dspark\":%s", ds4_engine_has_dspark(engine) ? "true" : "false");
    fprintf(out, ",\"quality\":%s", cfg->quality ? "true" : "false");
    fprintf(out, ",\"power_percent\":%d", ds4_engine_power(engine));
    fprintf(out, ",\"ctx_alloc\":%d", cfg->ctx_alloc);
    fprintf(out, ",\"prompt_file\":");
    json_write_string(out, run->prompt_path);
    fprintf(out, ",\"chat_prompt_file\":");
    json_write_string(out, run->chat_prompt_path);
    fprintf(out, ",\"system\":");
    json_write_string(out, run->system);
    fprintf(out, ",\"prompt_tokens\":%d", tokens ? tokens->len : 0);
    fprintf(out, ",\"frontier_tokens\":%d", run->frontier_tokens);
    fprintf(out, ",\"gen_tokens_requested\":%d", run->gen_tokens);
    fprintf(out, ",\"temperature\":%.9g", run->temperature);
    fprintf(out, ",\"top_k\":%d", run->top_k);
    fprintf(out, ",\"top_p\":%.9g", run->top_p);
    fprintf(out, ",\"min_p\":%.9g", run->min_p);
    fprintf(out, ",\"seed\":%llu", (unsigned long long)run->seed);
    fprintf(out, ",\"exclude_eos\":%s", run->exclude_eos ? "true" : "false");
    fprintf(out, ",\"prefill_ms\":%.6f", res->prefill_ms);
    fprintf(out, ",\"snapshot_ms\":%.6f", res->snapshot_ms);
    fprintf(out, ",\"decode_ms\":%.6f", res->decode_ms);
    fprintf(out, ",\"restore_ms\":%.6f", res->restore_ms);
    fprintf(out, ",\"tokens_emitted\":%d", res->emitted_tokens);
    fprintf(out, ",\"cycles\":%d", res->cycles);
    fprintf(out, ",\"accepted_total\":%d", res->accepted_total);
    fprintf(out, ",\"accepted_max\":%d", res->accepted_max);
    fprintf(out, ",\"accepted_mean\":%.9g",
            res->cycles > 0 ? (double)res->accepted_total / (double)res->cycles : 0.0);
    fprintf(out, ",\"tokens_per_second\":%.9g",
            res->decode_ms > 0.0 ? (double)res->emitted_tokens * 1000.0 / res->decode_ms : 0.0);
    fprintf(out, ",\"dspark_metrics_present\":%s", res->dspark_metrics_present ? "true" : "false");
    fprintf(out, ",\"dspark_timing_enabled\":%s", res->dspark_timing_enabled ? "true" : "false");
    fprintf(out, ",\"schedule_batched\":%s", res->schedule_batched ? "true" : "false");
    fprintf(out, ",\"scheduled_verify\":%s", res->scheduled_verify ? "true" : "false");
    fprintf(out, ",\"schedule_batch_limit\":%d", res->schedule_batch_limit);
    fprintf(out, ",\"dspark_cycle_count\":%d", res->dspark_cycles.len);
    fprintf(out, ",\"dspark_schedule_batch_limit_max\":%d",
            metric_max_i32(&res->dspark_cycles, metric_field_schedule_batch_limit));
    fprintf(out, ",\"dspark_rows_computed_mean\":%.9g",
            metric_mean_i32(&res->dspark_cycles, metric_field_rows_computed));
    fprintf(out, ",\"dspark_drafted_mean\":%.9g",
            metric_mean_i32(&res->dspark_cycles, metric_field_drafted));
    fprintf(out, ",\"dspark_verify_n_mean\":%.9g",
            metric_mean_i32(&res->dspark_cycles, metric_field_verify_n));
    fprintf(out, ",\"dspark_verified_mean\":%.9g",
            metric_mean_i32(&res->dspark_cycles, metric_field_verified));
    fprintf(out, ",\"dspark_cycle_accepted_mean\":%.9g",
            metric_mean_i32(&res->dspark_cycles, metric_field_accepted));
    fprintf(out, ",\"dspark_cycle_accepted_max\":%d",
            metric_max_i32(&res->dspark_cycles, metric_field_accepted));
    fprintf(out, ",\"dspark_decode_ms_mean\":%.9g",
            metric_mean_f64(&res->dspark_cycles, metric_field_decode_ms));
    fprintf(out, ",\"dspark_draft_ms_mean\":%.9g",
            metric_mean_f64(&res->dspark_cycles, metric_field_draft_ms));
    fprintf(out, ",\"dspark_verify_ms_mean\":%.9g",
            metric_mean_f64(&res->dspark_cycles, metric_field_verify_ms));
    fprintf(out, ",\"dspark_total_ms_mean\":%.9g",
            metric_mean_f64(&res->dspark_cycles, metric_field_total_ms));
    fprintf(out, ",\"dspark_verify_decode_ms_mean\":%.9g",
            metric_mean_f64(&res->dspark_cycles, metric_field_verify_decode_ms));
    fprintf(out, ",\"dspark_push_init_ms_mean\":%.9g",
            metric_mean_f64(&res->dspark_cycles, metric_field_push_init_ms));
    fprintf(out, ",\"dspark_push_verify_ms_mean\":%.9g",
            metric_mean_f64(&res->dspark_cycles, metric_field_push_verify_ms));
    fprintf(out, ",\"dspark_logits_read_ms_mean\":%.9g",
            metric_mean_f64(&res->dspark_cycles, metric_field_logits_read_ms));
    fprintf(out, ",\"dspark_cycles\":[");
    for (int i = 0; i < res->dspark_cycles.len; i++) {
        const ds4_dspark_cycle_metrics *m = &res->dspark_cycles.v[i];
        if (i) fprintf(out, ",");
        fprintf(out,
                "{\"scheduled_verify\":%s,\"batched_schedule\":%s,\"schedule_batch_limit\":%d,\"rows_computed\":%d,\"drafted\":%d,\"verify_n\":%d,\"verified\":%d,\"accepted\":%d,\"decode_ms\":%.6f,\"draft_ms\":%.6f,\"verify_ms\":%.6f,\"total_ms\":%.6f,\"pushes_init\":%d,\"pushes_verify\":%d,\"push_init_ms\":%.6f,\"push_verify_ms\":%.6f,\"verify_decode_ms\":%.6f,\"logits_read_ms\":%.6f,\"conf_logits\":[%.6f,%.6f,%.6f,%.6f,%.6f]",
                m->scheduled_verify ? "true" : "false",
                m->batched_schedule ? "true" : "false",
                m->schedule_batch_limit,
                m->rows_computed,
                m->drafted,
                m->verify_n,
                m->verified,
                m->accepted,
                m->decode_ms,
                m->draft_ms,
                m->verify_ms,
                m->total_ms,
                m->pushes_init,
                m->pushes_verify,
                m->push_init_ms,
                m->push_verify_ms,
                m->verify_decode_ms,
                m->logits_read_ms,
                m->conf_logits[0],
                m->conf_logits[1],
                m->conf_logits[2],
                m->conf_logits[3],
                m->conf_logits[4]);
        if (m->verify_dist.present) {
            fprintf(out,
                    ",\"verify_dist\":{\"n_compared\":%d,\"argmax_flips\":%d,\"max_abs_logit_diff\":%.6g,\"mean_tv\":%.6g,\"mean_kl_seq_batched\":%.6g,\"batched_verify_ms\":%.6f}",
                    m->verify_dist.n_compared,
                    m->verify_dist.argmax_flips,
                    m->verify_dist.max_abs_logit_diff,
                    m->verify_dist.mean_tv,
                    m->verify_dist.mean_kl_seq_batched,
                    m->verify_dist.batched_verify_ms);
        }
        fprintf(out,
                ",\"anchor_id\":%d,\"draft_ids\":[%d,%d,%d,%d,%d]",
                m->anchor_id,
                m->draft_ids[0], m->draft_ids[1], m->draft_ids[2], m->draft_ids[3], m->draft_ids[4]);
        fprintf(out, "}");
    }
    fprintf(out, "]");
    fprintf(out, ",\"eos_hit\":%s", res->eos_hit ? "true" : "false");
    fprintf(out, ",\"routed_quant_bits\":%d", ds4_engine_routed_quant_bits(engine));
    fprintf(out, ",\"error\":");
    json_write_string(out, res->ok ? NULL : res->err);
    fprintf(out, "}\n");
    fflush(out);
}

int main(int argc, char **argv) {
    spec_bench_config cfg = parse_options(argc, argv);
    spec_run_vec runs = load_runs(&cfg);
    if (cfg.ctx_alloc == 0) cfg.ctx_alloc = auto_ctx_alloc(&runs);

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .mtp_path = cfg.mtp_path,
        .dspark_path = cfg.dspark_path,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .prefill_chunk = cfg.prefill_chunk,
        .expert_profile_path = cfg.expert_profile_path,
        .power_percent = cfg.power_percent,
        .ssd_streaming_cache_experts = cfg.ssd_streaming_cache_experts,
        .ssd_streaming_cache_bytes = cfg.ssd_streaming_cache_bytes,
        .ssd_streaming_preload_experts = cfg.ssd_streaming_preload_experts,
        .simulate_used_memory_bytes = cfg.simulate_used_memory_bytes,
        .warm_weights = cfg.warm_weights,
        .quality = cfg.quality,
        .ssd_streaming = cfg.ssd_streaming,
        .ssd_streaming_cold = cfg.ssd_streaming_cold,
    };

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) {
        for (int i = 0; i < runs.len; i++) run_free(&runs.v[i]);
        free(runs.v);
        return 1;
    }

    FILE *out = stdout;
    if (cfg.jsonl_path) {
        out = fopen(cfg.jsonl_path, "wb");
        if (!out) {
            fprintf(stderr, "ds4-spec-bench: failed to open %s: %s\n", cfg.jsonl_path, strerror(errno));
            ds4_engine_close(engine);
            for (int i = 0; i < runs.len; i++) run_free(&runs.v[i]);
            free(runs.v);
            return 1;
        }
    }

    prompt_cache cache = {0};
    int rc = 0;
    bool did_rewrite = false;
    if (cfg.rewrite_frontier_path) {
        rc = rewrite_frontier_config(&runs, &cache, engine, cfg.rewrite_frontier_path);
        did_rewrite = true;
    } else {
        const int invalid = validate_frontier_runs(&runs, &cache, engine);
        if (invalid > 0) {
            fprintf(stderr, "ds4-spec-bench: REFUSING to run — %d of %d run(s) have invalid frontier>prompt. "
                            "Fix the config (set frontier_tokens <= prompt length for every entry) or run with "
                            "--rewrite-frontier OUT to emit a corrected config, then re-run. Aborting (no silent skips).\n",
                    invalid, runs.len);
            rc = 2;
        }
    }
    if (cfg.dump_logprobs_jsonl) setenv("DS4_BENCH_DUMP_LOGPROBS_JSONL", cfg.dump_logprobs_jsonl, 1);
    else unsetenv("DS4_BENCH_DUMP_LOGPROBS_JSONL");
    if (cfg.logprobs_top_k > 0) {
        char kb[16];
        snprintf(kb, sizeof(kb), "%d", cfg.logprobs_top_k);
        setenv("DS4_BENCH_LOGPROBS_TOP_K", kb, 1);
    }
    for (int i = 0; rc == 0 && !did_rewrite && i < runs.len; i++) {
        spec_run *run = &runs.v[i];
        char err[256] = {0};
        if (cfg.dump_hidden_dir && run->id && run->id[0]) {
            char dpath[2048];
            snprintf(dpath, sizeof(dpath), "%s/%s.bin", cfg.dump_hidden_dir, run->id);
            setenv("DS4_DSPARK_DUMP_HIDDEN", dpath, 1);
        } else {
            unsetenv("DS4_DSPARK_DUMP_HIDDEN");
        }
        /* teacher_force: per-prompt forced-token input + IQ2-argmax output paths */
        if (run->mode == SPEC_MODE_TEACHER_FORCE) {
            if (cfg.force_tokens_dir && run->id && run->id[0]) {
                char fpath[2048];
                snprintf(fpath, sizeof(fpath), "%s/%s.tokens", cfg.force_tokens_dir, run->id);
                setenv("DS4_FORCE_TOKENS_PATH", fpath, 1);
            } else {
                unsetenv("DS4_FORCE_TOKENS_PATH");
            }
            if (cfg.dump_hidden_dir && run->id && run->id[0]) {
                char apath[2048];
                snprintf(apath, sizeof(apath), "%s/%s.iq2argmax", cfg.dump_hidden_dir, run->id);
                setenv("DS4_IQ2_ARGMAX_PATH", apath, 1);
            } else {
                unsetenv("DS4_IQ2_ARGMAX_PATH");
            }
        }
        const ds4_tokens *tokens = prompt_cache_get(&cache, engine, run, err, sizeof(err));
        run_result res = {0};
        if (!tokens) {
            snprintf(res.err, sizeof(res.err), "%s", err);
        } else if (run->mode == SPEC_MODE_SPECULATIVE_ARGMAX &&
                   !ds4_engine_has_dspark(engine) && !ds4_engine_has_mtp(engine)) {
            snprintf(res.err, sizeof(res.err), "speculative_argmax requested but engine has no speculative drafter");
        } else if (run->mode == SPEC_MODE_TEACHER_FORCE &&
                   (!ds4_engine_has_dspark(engine) || !cfg.dump_hidden_dir || !cfg.force_tokens_dir)) {
            snprintf(res.err, sizeof(res.err), "teacher_force requires dspark loaded + --dump-hidden-dir + --force-tokens-dir");
        } else {
            res = execute_run(engine, cfg.ctx_alloc, run, tokens);
        }
        if (!res.ok) rc = 1;
        write_result_jsonl(out, &cfg, engine, run, tokens, &res);
        run_result_free(&res);
    }

    for (int i = 0; i < cache.len; i++) {
        free(cache.v[i].prompt_path);
        free(cache.v[i].chat_prompt_path);
        free(cache.v[i].system);
        ds4_tokens_free(&cache.v[i].tokens);
    }
    free(cache.v);

    if (out != stdout) fclose(out);
    ds4_engine_close(engine);
    for (int i = 0; i < runs.len; i++) run_free(&runs.v[i]);
    free(runs.v);
    return rc;
}
