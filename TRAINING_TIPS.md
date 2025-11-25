# Three-Legged Walking Training Tips

## Problem: Robot Standing Still Instead of Walking

### Root Causes Identified:
1. **Poor Command Distribution**: Random commands [-1,1] often included zero/low velocities
2. **Reward Structure Favoring Inaction**: Penalties outweighed motion rewards
3. **Conditional Air Time Rewards**: Only rewarded when commanded velocity > 0.1
4. **Insufficient Forward Motion Incentive**: No direct reward for forward progress

### Solutions Implemented:

#### 1. Command Generation Improvements
- **Before**: `uniform(-1.0, 1.0)` for all commands
- **After**: 
  - Forward velocity: `uniform(0.2, 1.5)` m/s (always forward)
  - Lateral velocity: `uniform(-0.3, 0.3)` m/s (limited side motion)
  - Yaw rate: `uniform(-0.5, 0.5)` rad/s (limited turning)

#### 2. Reward Scale Rebalancing
- **Linear Velocity Tracking**: 1.0 → 3.0 (3x stronger forward motion reward)
- **Forward Progress**: NEW 2.0 (direct reward for forward velocity)
- **Feet Air Time**: 0.8 → 1.5 (encourage stepping)
- **Joint Torque Penalty**: -2.5e-5 → -1.0e-5 (reduced to allow walking torques)
- **Action Rate Penalty**: -0.01 → -0.005 (allow more dynamic actions)
- **Flat Orientation**: -3.0 → -1.0 (three-legged robot needs to tilt)

#### 3. Training Hyperparameter Improvements
- **Entropy Coefficient**: 0.01 → 0.02 (more exploration)
- **Learning Epochs**: 5 → 8 (more learning per batch)
- **Learning Rate**: 1e-3 → 3e-4 (more stable learning)
- **Max Iterations**: 1500 → 2000 (longer training)

#### 4. Initial Pose Optimization
- **Height**: 0.6 → 0.55 m (lower, more stable)
- **Hip Angles**: Added asymmetric abduction for stability
- **Joint Positions**: Optimized for walking readiness

### Expected Training Behavior:

#### Phase 1 (Iterations 0-500): Learning Balance
- Robot learns to maintain balance with 3 legs
- May still fall or struggle with basic stability
- Forward progress rewards start taking effect

#### Phase 2 (Iterations 500-1000): Basic Walking
- Robot develops basic stepping patterns
- Begins moving forward consistently
- Still may have irregular gait

#### Phase 3 (Iterations 1000-1500): Gait Refinement
- Develops more regular three-legged gait
- Better velocity tracking
- Improved energy efficiency

#### Phase 4 (Iterations 1500-2000): Optimization
- Fine-tunes gait for different speeds
- Better turning and lateral motion
- Robust walking behavior

### Monitoring Training Progress:

#### Key Metrics to Watch:
1. **Episode_Reward/track_lin_vel_xy_exp**: Should increase steadily
2. **Episode_Reward/forward_progress**: Should become positive and grow
3. **Episode_Reward/feet_air_time**: Should increase as stepping improves
4. **Base Velocity**: Monitor actual forward velocity in logs

#### Troubleshooting:

**If robot still stands still after 500 iterations:**
- Increase `forward_progress_reward_scale` to 3.0 or 4.0
- Reduce `stability_reward_scale` to 0.2
- Check that commands are being generated correctly

**If robot falls frequently:**
- Increase `stability_reward_scale` to 1.0
- Reduce `lin_vel_reward_scale` to 2.0 temporarily
- Check initial joint positions

**If training is unstable:**
- Reduce learning rate to 1e-4
- Increase `flat_orientation_reward_scale` penalty (more negative)
- Add gradient clipping if not already present

### Command Line Training:
```bash
# Start fresh training with new rewards
python scripts/rsl_rl/train.py --task=less_leg_walking_1:LessLegWalkingFlatEnvCfg --headless

# Resume from checkpoint (if needed)
python scripts/rsl_rl/train.py --task=less_leg_walking_1:LessLegWalkingFlatEnvCfg --resume --headless

# Monitor training progress
tensorboard --logdir logs/rsl_rl/less_leg_walking_flat
```

### Additional Improvements for Future Iterations:

1. **Adaptive Command Curriculum**: Start with slower speeds, gradually increase
2. **Gait-Specific Rewards**: Reward specific three-legged gait patterns
3. **Energy Efficiency**: Add specific energy optimization rewards
4. **Robustness Training**: Add external disturbances during training
5. **Multi-Speed Training**: Train on different velocity ranges simultaneously
