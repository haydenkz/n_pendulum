#include "n_pendulum.h"

int main(int argc, char** argv) {
    float observations[DP_OBS_SIZE] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    int num_links = argc > 1 ? atoi(argv[1]) : DP_DEFAULT_LINKS;

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
        .cart_mass = 1.0f,
        .link_mass = 0.1f,
        .link_length = 0.7f,
        .gravity = 9.8f,
        .force_mag = 10.0f,
        .dt = 0.02f,
    };

    init(&env);
    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        int center_action = (DP_ACTIONS - 1) / 2;
        if (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT)) {
            if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) actions[0] = 0;
            else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) actions[0] = DP_ACTIONS - 1;
            else actions[0] = center_action;
        } else if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) actions[0] = center_action - 2;
        else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) actions[0] = center_action + 2;
        else actions[0] = center_action;
        c_step(&env);
        c_render(&env);
    }
    c_close(&env);
    return 0;
}
