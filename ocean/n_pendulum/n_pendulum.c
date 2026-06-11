#include <stdio.h>
#include <time.h>
#include "n_pendulum.h"
#include "puffernet.h"

int main(int argc, char** argv) {
    float observations[DP_OBS_SIZE] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    int num_links = argc > 1 ? atoi(argv[1]) : DP_DEFAULT_LINKS;
    const char* weights_path = argc > 2 ? argv[2] : NULL;
    int hidden_size = argc > 3 ? atoi(argv[3]) : 100;
    int num_layers = argc > 4 ? atoi(argv[4]) : 4;

    NPendulum env = {
        .observations = observations,
        .actions = actions,
        .rewards = rewards,
        .terminals = terminals,
        .num_agents = 1,
        .rng = 1,
        .num_links = num_links,
        .min_episode_steps = DP_DEFAULT_MIN_STEPS,
        .max_episode_steps = DP_DEFAULT_MAX_STEPS,
        .upright_reset_prob = 0.15f,
        .upright_angle_noise = 0.15f,
        .upright_vel_noise = 0.2f,
        .delta_scale = DP_DEFAULT_DELTA_SCALE,
        .stable_height = DP_DEFAULT_STABLE_HEIGHT,
        .stable_vel = DP_DEFAULT_STABLE_VEL,
        .hold_ramp_steps = DP_DEFAULT_HOLD_RAMP_STEPS,
        .stable_bonus = DP_DEFAULT_STABLE_BONUS,
        .hold_bonus = DP_DEFAULT_HOLD_BONUS,
        .cart_mass = 1.0f,
        .link_mass = 0.1f,
        .link_length = 0.7f,
        .gravity = 9.8f,
        .force_mag = 10.0f,
        .dt = 0.02f,
    };

    srand(time(NULL));
    Weights* weights = NULL;
    PufferNet* net = NULL;
    if (weights_path != NULL) {
        weights = load_weights(weights_path);
        if (weights == NULL) {
            fprintf(stderr, "Failed to load weights from %s\n", weights_path);
            return 1;
        }
        int logit_sizes[1] = {DP_ACTIONS};
        net = make_puffernet(weights, 1, DP_OBS_SIZE, hidden_size, num_layers, logit_sizes, 1);
    }

    init(&env);
    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        bool manual_control = net == NULL || IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
        int center_action = (DP_ACTIONS - 1) / 2;
        if (!manual_control) {
            forward_puffernet(net, observations, actions);
        } else if (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT)) {
            if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) actions[0] = 0;
            else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) actions[0] = DP_ACTIONS - 1;
            else actions[0] = center_action;
        } else if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) actions[0] = center_action - 2;
        else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) actions[0] = center_action + 2;
        else actions[0] = center_action;
        c_step(&env);
        c_render(&env);
    }
    if (net != NULL) free_puffernet(net);
    if (weights != NULL) free(weights);
    c_close(&env);
    return 0;
}
