// N-pendulum swing-up and balance task with discrete cart forces.

#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

#ifndef DP_MAX_LINKS
#define DP_MAX_LINKS 7
#endif

#define DP_DEFAULT_LINKS 2
#define DP_OBS_FEATURES_PER_LINK 6
#define DP_OBS_SIZE (2 + DP_OBS_FEATURES_PER_LINK * DP_MAX_LINKS)
#define DP_ACTIONS 13
#define DP_DEFAULT_MIN_STEPS 1000
#define DP_DEFAULT_MAX_STEPS 2000
#define DP_X_THRESHOLD 5.0f
#define DP_WIDTH 1200
#define DP_HEIGHT 720
#define DP_MAX_DOF (DP_MAX_LINKS + 1)
#define DP_MAX_REWARD 1.0f

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float x_threshold_termination;
    float max_steps_termination;
    float hold_time;
    float n;
} Log;

typedef struct NPendulum {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;
    Log log;

    int num_links;
    float x;
    float x_dot;
    float theta[DP_MAX_LINKS];
    float theta_dot[DP_MAX_LINKS];
    float prev_height;
    int tick;
    float episode_return;
    int upright_steps;
    int max_upright_steps;
    int min_episode_steps;
    int max_episode_steps;
    int episode_max_steps;
    float upright_reset_prob;
    float upright_angle_noise;
    float upright_vel_noise;

    float cart_mass;
    float link_mass;
    float link_length;
    float gravity;
    float force_mag;
    float dt;
} NPendulum;

typedef NPendulum DoublePendulum;

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 255};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_YELLOW = (Color){245, 197, 66, 255};

static inline float dp_randf(NPendulum* env, float lo, float hi) {
    float t = (float)rand_r(&env->rng) / (float)RAND_MAX;
    return lo + t * (hi - lo);
}

static inline int dp_randi(NPendulum* env, int lo, int hi) {
    if (hi <= lo) return lo;
    return lo + (int)(rand_r(&env->rng) % (unsigned int)(hi - lo + 1));
}

static inline float wrap_pi(float x) {
    while (x > M_PI) x -= 2.0f * M_PI;
    while (x < -M_PI) x += 2.0f * M_PI;
    return x;
}

static inline float clamp01(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

static inline float track_limit(NPendulum* env) {
    float total_length = env->link_length * (float)env->num_links;
    return fmaxf(DP_X_THRESHOLD, 2.0f * total_length);
}

static inline int clamp_num_links(int num_links) {
    if (num_links < 1) return DP_DEFAULT_LINKS;
    if (num_links > DP_MAX_LINKS) return DP_MAX_LINKS;
    return num_links;
}

void compute_observations(NPendulum* env) {
    float x_limit = track_limit(env);
    env->observations[0] = env->x / x_limit;
    env->observations[1] = env->x_dot / 5.0f;
    for (int i = 0; i < env->num_links; i++) {
        int off = 2 + DP_OBS_FEATURES_PER_LINK * i;
        float rel_theta = i == 0 ? env->theta[i] : env->theta[i] - env->theta[i - 1];
        float rel_theta_dot = i == 0
            ? env->theta_dot[i]
            : env->theta_dot[i] - env->theta_dot[i - 1];
        env->observations[off] = sinf(env->theta[i]);
        env->observations[off + 1] = cosf(env->theta[i]);
        env->observations[off + 2] = env->theta_dot[i] / 8.0f;
        env->observations[off + 3] = sinf(rel_theta);
        env->observations[off + 4] = cosf(rel_theta);
        env->observations[off + 5] = rel_theta_dot / 8.0f;
    }
    int used = 2 + DP_OBS_FEATURES_PER_LINK * env->num_links;
    if (used < DP_OBS_SIZE) {
        memset(env->observations + used, 0,
            (DP_OBS_SIZE - used) * sizeof(float));
    }
}

float pendulum_height(NPendulum* env) {
    float tip_y = 0.0f;
    float max_y = 0.0f;
    for (int i = 0; i < env->num_links; i++) {
        tip_y += env->link_length * cosf(env->theta[i]);
        max_y += env->link_length;
    }
    return 0.5f * (tip_y / max_y + 1.0f);
}

static inline float link_height(float theta) {
    return 0.5f * (cosf(theta) + 1.0f);
}

void add_log(NPendulum* env, bool x_done, bool timeout) {
    env->log.perf += clamp01(env->prev_height);
    env->log.score += env->episode_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->tick;
    env->log.x_threshold_termination += x_done ? 1.0f : 0.0f;
    env->log.max_steps_termination += timeout ? 1.0f : 0.0f;
    env->log.hold_time += (float)env->max_upright_steps;
    env->log.n += 1.0f;
}

void init(NPendulum* env) {
    env->num_agents = 1;
    env->num_links = clamp_num_links(env->num_links);
    if (env->min_episode_steps <= 0) env->min_episode_steps = DP_DEFAULT_MIN_STEPS;
    if (env->max_episode_steps <= 0) env->max_episode_steps = DP_DEFAULT_MAX_STEPS;
    if (env->min_episode_steps > env->max_episode_steps) {
        int tmp = env->min_episode_steps;
        env->min_episode_steps = env->max_episode_steps;
        env->max_episode_steps = tmp;
    }
    env->upright_reset_prob = clamp01(env->upright_reset_prob);
    if (env->upright_angle_noise < 0.0f) env->upright_angle_noise = 0.0f;
    if (env->upright_vel_noise < 0.0f) env->upright_vel_noise = 0.0f;
    if (!(env->cart_mass > 0.0f)) env->cart_mass = 1.0f;
    if (!(env->link_mass > 0.0f)) env->link_mass = 0.1f;
    if (!(env->link_length > 0.0f)) env->link_length = 0.7f;
    if (!(env->gravity > 0.0f)) env->gravity = 9.8f;
    if (!(env->force_mag > 0.0f)) env->force_mag = 10.0f;
    if (!(env->dt > 0.0f)) env->dt = 0.02f;
}

void c_reset(NPendulum* env) {
    env->x = dp_randf(env, -0.04f, 0.04f);
    env->x_dot = dp_randf(env, -0.04f, 0.04f);
    bool upright_start = dp_randf(env, 0.0f, 1.0f) < env->upright_reset_prob;
    if (upright_start) {
        for (int i = 0; i < env->num_links; i++) {
            env->theta[i] = dp_randf(env,
                -env->upright_angle_noise, env->upright_angle_noise);
            env->theta_dot[i] = dp_randf(env,
                -env->upright_vel_noise, env->upright_vel_noise);
        }
    } else {
        for (int i = 0; i < env->num_links; i++) {
            env->theta[i] = M_PI + dp_randf(env, -0.08f, 0.08f);
            env->theta_dot[i] = dp_randf(env, -0.04f, 0.04f);
        }
    }
    env->tick = 0;
    env->episode_max_steps = dp_randi(env,
        env->min_episode_steps, env->max_episode_steps);
    env->episode_return = 0.0f;
    env->upright_steps = 0;
    env->max_upright_steps = 0;
    env->prev_height = pendulum_height(env);
    compute_observations(env);
}

static bool solve_linear(int n, float A[DP_MAX_DOF][DP_MAX_DOF],
        float b[DP_MAX_DOF], float x[DP_MAX_DOF]) {
    for (int i = 0; i < n; i++) {
        int pivot = i;
        float best = fabsf(A[i][i]);
        for (int r = i + 1; r < n; r++) {
            float v = fabsf(A[r][i]);
            if (v > best) {
                best = v;
                pivot = r;
            }
        }
        if (best < 1e-8f) {
            memset(x, 0, n * sizeof(float));
            return false;
        }
        if (pivot != i) {
            for (int c = i; c < n; c++) {
                float tmp = A[i][c];
                A[i][c] = A[pivot][c];
                A[pivot][c] = tmp;
            }
            float tmp = b[i];
            b[i] = b[pivot];
            b[pivot] = tmp;
        }

        float inv = 1.0f / A[i][i];
        for (int c = i; c < n; c++) A[i][c] *= inv;
        b[i] *= inv;
        for (int r = 0; r < n; r++) {
            if (r == i) continue;
            float f = A[r][i];
            for (int c = i; c < n; c++) A[r][c] -= f * A[i][c];
            b[r] -= f * b[i];
        }
    }
    for (int i = 0; i < n; i++) x[i] = b[i];
    return true;
}

void integrate_physics(NPendulum* env, float force) {
    int n = env->num_links;
    int dof = n + 1;
    float l = env->link_length;
    float m = env->link_mass;
    float tail_mass[DP_MAX_LINKS];
    float A[DP_MAX_DOF][DP_MAX_DOF];
    float b[DP_MAX_DOF];
    float qdd[DP_MAX_DOF];

    for (int i = 0; i < n; i++) {
        tail_mass[i] = (float)(n - i) * m;
    }

    A[0][0] = env->cart_mass + tail_mass[0];
    b[0] = force;
    for (int i = 0; i < n; i++) {
        float si = sinf(env->theta[i]);
        float ci = cosf(env->theta[i]);
        float wi = env->theta_dot[i];
        A[0][i + 1] = tail_mass[i] * l * ci;
        A[i + 1][0] = A[0][i + 1];
        b[0] += tail_mass[i] * l * si * wi * wi;
    }

    for (int i = 0; i < n; i++) {
        float ti = env->theta[i];
        b[i + 1] = tail_mass[i] * env->gravity * l * sinf(ti);
        for (int j = 0; j < n; j++) {
            int tail_idx = i > j ? i : j;
            float coupled_mass = tail_mass[tail_idx];
            float tj = env->theta[j];
            A[i + 1][j + 1] = coupled_mass * l * l * cosf(ti - tj);
            if (i != j) {
                float wj = env->theta_dot[j];
                b[i + 1] -= coupled_mass * l * l * sinf(ti - tj) * wj * wj;
            }
        }
    }

    if (!solve_linear(dof, A, b, qdd)) {
        return;
    }

    env->x_dot += env->dt * qdd[0];
    env->x_dot = fminf(fmaxf(env->x_dot, -20.0f), 20.0f);
    env->x += env->dt * env->x_dot;
    for (int i = 0; i < n; i++) {
        env->theta_dot[i] += env->dt * qdd[i + 1];
        env->theta_dot[i] = fminf(fmaxf(env->theta_dot[i], -30.0f), 30.0f);
        env->theta[i] = wrap_pi(env->theta[i] + env->dt * env->theta_dot[i]);
    }
}

float upright_reward(NPendulum* env, float force) {
    (void)force;
    int n = env->num_links;
    float height = 0.0f;
    float tip_x = 0.0f;
    float avg_abs_vel = 0.0f;
    float max_abs_vel = 0.0f;
    for (int i = 0; i < n; i++) {
        float theta = wrap_pi(env->theta[i]);
        float h = link_height(theta);
        float w = fabsf(env->theta_dot[i]);
        height += h;
        avg_abs_vel += w;
        max_abs_vel = fmaxf(max_abs_vel, w);
        tip_x += env->link_length * sinf(theta);
    }
    height /= (float)n;
    avg_abs_vel /= (float)n;

    float x_limit = track_limit(env);
    float total_length = env->link_length * (float)n;
    float height_delta = height - env->prev_height;
    float delta_reward = fminf(fmaxf(height_delta, -1.0f), 1.0f);
    float cart_center = 1.0f - clamp01(fabsf(env->x) / x_limit);
    float tip_center = 1.0f - clamp01(fabsf(tip_x) / fmaxf(0.5f * total_length, 0.001f));
    float slow = 1.0f - clamp01(avg_abs_vel / 6.0f);
    float balance_quality = height * slow * cart_center * tip_center;

    bool stable = height > 0.9f
        && max_abs_vel < 1.5f
        && fabsf(env->x_dot) < 1.0f
        && cart_center > 0.5f
        && tip_center > 0.5f;
    if (stable) env->upright_steps += 1;
    else env->upright_steps = 0;
    if (env->upright_steps > env->max_upright_steps) {
        env->max_upright_steps = env->upright_steps;
    }

    float hold_progress = clamp01((float)env->upright_steps / 100.0f);
    float reward = delta_reward
        + 0.02f * height * cart_center
        + 0.03f * balance_quality
        + (stable ? 0.05f + 0.15f * hold_progress : 0.0f);
    env->prev_height = height;
    return fminf(fmaxf(reward, -DP_MAX_REWARD), DP_MAX_REWARD);
}

static bool invalid_state(NPendulum* env) {
    if (!isfinite(env->x) || !isfinite(env->x_dot)) return true;
    for (int i = 0; i < env->num_links; i++) {
        if (!isfinite(env->theta[i]) || !isfinite(env->theta_dot[i])) return true;
    }
    return false;
}

void c_step(NPendulum* env) {
    float a = env->actions[0];
    if (!isfinite(a)) a = 1.0f;
    int action = (int)a;
    int center_action = (DP_ACTIONS - 1) / 2;
    if ((unsigned)action >= DP_ACTIONS) action = center_action;
    float force_scale = (float)(action - center_action) / (float)center_action;
    float force = force_scale * env->force_mag;

    integrate_physics(env, force);
    env->tick += 1;

    bool invalid = invalid_state(env);
    float x_limit = track_limit(env);
    bool x_done = env->x < -x_limit || env->x > x_limit;
    bool timeout = env->tick >= env->episode_max_steps;
    bool done = invalid || x_done || timeout;
    env->rewards[0] = upright_reward(env, force);
    if (invalid || x_done) {
        env->rewards[0] = -DP_MAX_REWARD;
    }
    env->episode_return += env->rewards[0];
    env->terminals[0] = (invalid || x_done) ? 1.0f : 0.0f;

    if (done) {
        add_log(env, invalid || x_done, timeout);
        c_reset(env);
        return;
    }
    compute_observations(env);
}

void c_render(NPendulum* env) {
    if (!IsWindowReady()) {
        InitWindow(DP_WIDTH, DP_HEIGHT, "PufferLib N-Pendulum");
        SetTargetFPS(30);
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    if (IsKeyPressed(KEY_TAB)) ToggleFullscreen();
    if (invalid_state(env)) return;

    float rail_y = DP_HEIGHT * 0.50f;
    float total_length = env->link_length * (float)env->num_links;
    float x_limit = track_limit(env);
    float pixels_per_meter = (DP_WIDTH - 80.0f) / (2.0f * x_limit);
    float max_chain_pixels = fminf(rail_y, DP_HEIGHT - rail_y) - 24.0f;
    pixels_per_meter = fminf(pixels_per_meter,
        max_chain_pixels / fmaxf(total_length, 0.001f));

    float cart_x = DP_WIDTH / 2.0f + env->x * pixels_per_meter;
    cart_x = fminf(fmaxf(cart_x, 32.0f), DP_WIDTH - 32.0f);
    float cart_y = rail_y - 16.0f;
    float link_pixels = env->link_length * pixels_per_meter;
    Vector2 prev = {cart_x, cart_y};
    Color colors[] = {PUFF_RED, PUFF_YELLOW, PUFF_CYAN};

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    int rail_min = (int)(DP_WIDTH / 2.0f - x_limit * pixels_per_meter);
    int rail_max = (int)(DP_WIDTH / 2.0f + x_limit * pixels_per_meter);
    DrawLine(rail_min, (int)rail_y, rail_max, (int)rail_y, PUFF_CYAN);
    DrawRectangle((int)(cart_x - 28), (int)(cart_y - 12), 56, 24, PUFF_CYAN);
    DrawCircleV(prev, 8.0f, PUFF_WHITE);
    for (int i = 0; i < env->num_links; i++) {
        Vector2 next = {
            prev.x + sinf(env->theta[i]) * link_pixels,
            prev.y - cosf(env->theta[i]) * link_pixels,
        };
        float thickness = fmaxf(3.0f, 7.0f - 0.25f * (float)i);
        DrawLineEx(prev, next, thickness, colors[i % 3]);
        DrawCircleV(next, i == env->num_links - 1 ? 10.0f : 7.0f, PUFF_WHITE);
        prev = next;
    }
    DrawText(TextFormat("links %d  steps %d/%d  return %.1f  hold %d/%d",
        env->num_links, env->tick, env->episode_max_steps, env->episode_return,
        env->upright_steps, env->max_upright_steps),
        20, 20, 20, PUFF_WHITE);
    DrawText(TextFormat("x %.2f  first %.1f  last %.1f",
        env->x, env->theta[0] * 180.0f / M_PI,
        env->theta[env->num_links - 1] * 180.0f / M_PI),
        20, 48, 20, PUFF_WHITE);
    EndDrawing();
}

void c_close(NPendulum* env) {
    (void)env;
    if (IsWindowReady()) {
        CloseWindow();
    }
}
