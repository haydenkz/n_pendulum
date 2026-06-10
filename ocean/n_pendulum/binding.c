#include "n_pendulum.h"

#define OBS_SIZE DP_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {DP_ACTIONS}
#define OBS_TENSOR_T FloatTensor

#define Env NPendulum
#include "vecenv.h"

static float kwarg_or(Dict* kwargs, const char* key, float fallback) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    return item == NULL ? fallback : (float)item->value;
}

static int kwarg_int_or(Dict* kwargs, const char* key, int fallback) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    return item == NULL ? fallback : (int)item->value;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->num_links = kwarg_int_or(kwargs, "num_links", DP_DEFAULT_LINKS);
    env->min_episode_steps = kwarg_int_or(kwargs, "min_episode_steps",
        kwarg_int_or(kwargs, "max_steps", DP_DEFAULT_MIN_STEPS));
    env->max_episode_steps = kwarg_int_or(kwargs, "max_episode_steps",
        kwarg_int_or(kwargs, "max_steps", DP_DEFAULT_MAX_STEPS));
    env->upright_reset_prob = kwarg_or(kwargs, "upright_reset_prob", 0.0f);
    env->upright_angle_noise = kwarg_or(kwargs, "upright_angle_noise", 0.15f);
    env->upright_vel_noise = kwarg_or(kwargs, "upright_vel_noise", 0.2f);
    env->cart_mass = kwarg_or(kwargs, "cart_mass", 1.0f);
    env->link_mass = kwarg_or(kwargs, "link_mass",
        kwarg_or(kwargs, "link1_mass", 0.1f));
    env->link_length = kwarg_or(kwargs, "link_length",
        kwarg_or(kwargs, "link1_length", 0.5f));
    env->gravity = kwarg_or(kwargs, "gravity", 9.8f);
    env->force_mag = kwarg_or(kwargs, "force_mag", 10.0f);
    env->dt = kwarg_or(kwargs, "dt", 0.02f);
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "perf", log->perf);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "x_threshold_termination", log->x_threshold_termination);
    dict_set(out, "max_steps_termination", log->max_steps_termination);
    dict_set(out, "hold_time", log->hold_time);
    dict_set(out, "n", log->n);
}
