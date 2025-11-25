# Three-Legged Robot Walking Task

## Overview

This task implements a three-legged walking environment for the ANYmal-C quadrupedal robot using Isaac Lab. The robot is modified to walk with only three legs (Left Front, Left Hind, and Right Hind) while the Right Front leg is disabled and folded up to avoid ground contact.

## Key Features

- **Modified Robot Configuration**: ANYmal-C robot with disabled RF (Right Front) leg
- **Reduced Action Space**: 9 actions (3 legs × 3 joints) instead of 12
- **Adjusted Observations**: Observation space reduced to account for missing leg
- **Specialized Rewards**: Reward functions tuned for three-legged locomotion
- **Stability Focus**: Additional rewards for maintaining balance with asymmetric leg configuration

## Robot Configuration

### Active Legs
- **LF (Left Front)**: HAA, HFE, KFE joints
- **LH (Left Hind)**: HAA, HFE, KFE joints  
- **RH (Right Hind)**: HAA, HFE, KFE joints

### Disabled Leg
- **RF (Right Front)**: Folded up with fixed joint positions
  - RF_HAA: 0.0 (neutral)
  - RF_HFE: -1.2 (fold hip more)
  - RF_KFE: 2.0 (bend knee to tuck leg up)

## Action Space

- **Size**: 9 (reduced from 12)
- **Format**: [LF_HAA, LF_HFE, LF_KFE, LH_HAA, LH_HFE, LH_KFE, RH_HAA, RH_HFE, RH_KFE]
- **Range**: Scaled by `action_scale` parameter (default: 0.5)

## Observation Space

### Flat Terrain: 39 dimensions
- Root linear velocity (body frame): 3
- Root angular velocity (body frame): 3
- Projected gravity (body frame): 3
- Commands (x_vel, y_vel, yaw_rate): 3
- Joint positions (9 active joints): 9
- Joint velocities (9 active joints): 9
- Previous actions: 9

### Rough Terrain: 226 dimensions
- All flat terrain observations: 39
- Height map data: 187

## Reward Components

1. **Linear Velocity Tracking** (`lin_vel_reward_scale = 1.0`)
   - Exponential reward for following x/y velocity commands

2. **Yaw Rate Tracking** (`yaw_rate_reward_scale = 0.5`)
   - Exponential reward for following angular velocity commands

3. **Z Velocity Penalty** (`z_vel_reward_scale = -2.0`)
   - Penalize vertical motion

4. **Angular Velocity Penalty** (`ang_vel_reward_scale = -0.05`)
   - Penalize excessive roll/pitch motion

5. **Joint Torque Penalty** (`joint_torque_reward_scale = -2.5e-5`)
   - Encourage energy efficiency

6. **Joint Acceleration Penalty** (`joint_accel_reward_scale = -2.5e-7`)
   - Encourage smooth motion

7. **Action Rate Penalty** (`action_rate_reward_scale = -0.01`)
   - Penalize rapid action changes

8. **Feet Air Time Reward** (`feet_air_time_reward_scale = 0.8`)
   - Reward proper gait patterns with appropriate swing phases

9. **Undesired Contact Penalty** (`undesired_contact_reward_scale = -1.0`)
   - Penalize contacts on thigh segments

10. **Flat Orientation Penalty** (`flat_orientation_reward_scale = -3.0`)
    - Reduced penalty as three-legged robot may naturally tilt

11. **Stability Reward** (`stability_reward_scale = 2.0`)
    - New reward for maintaining balance with asymmetric configuration

## Environment Variants

### LessLegWalkingFlatEnvCfg
- Flat terrain locomotion
- 4096 parallel environments
- 20-second episodes
- Focus on basic three-legged walking

### LessLegWalkingRoughEnvCfg
- Rough terrain locomotion with height sensing
- Includes 187-dimensional height map observations
- More challenging environment for robust policy learning

## Training Configuration

### PPO Hyperparameters
- **Flat Terrain**:
  - Max iterations: 1500
  - Steps per environment: 24
  - Learning rate: 1e-3
  - Entropy coefficient: 0.01

- **Rough Terrain**:
  - Max iterations: 2000
  - Lower entropy coefficient: 0.005 (for more stable policy)

## Usage

1. **Install Isaac Lab** following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

2. **Install this extension**:
   ```bash
   python -m pip install -e source/less_leg_walking_1
   ```

3. **Train the robot**:
   ```bash
   # Flat terrain
   python scripts/rsl_rl/train.py --task=less_leg_walking_1:LessLegWalkingFlatEnvCfg

   # Rough terrain  
   python scripts/rsl_rl/train.py --task=less_leg_walking_1:LessLegWalkingRoughEnvCfg
   ```

4. **Test with random actions**:
   ```bash
   python scripts/random_agent.py --task less_leg_walking_1:LessLegWalkingFlatEnvCfg --num_envs 16
   ```

## Key Implementation Details

### Action Mapping
The 9-dimensional action vector is mapped to the robot's 12 joints by:
- Taking actions 0-2 for LF leg (joints 0, 4, 8)
- Taking actions 3-5 for LH leg (joints 1, 5, 9)  
- Taking actions 6-8 for RH leg (joints 3, 7, 11)
- Setting RF leg joints (2, 6, 10) to fixed folded positions

### Observation Filtering
Only active joint data is included in observations to maintain consistency with the reduced action space.

### Stability Considerations
- Increased feet air time reward to encourage stable gait patterns
- Reduced flat orientation penalty to account for natural asymmetric tilting
- Added stability reward based on angular velocity to promote balance

## Future Improvements

1. **Adaptive Gait Patterns**: Implement specialized three-legged gait controllers
2. **Dynamic Leg Selection**: Allow switching which leg is disabled during training
3. **Fault-Tolerant Locomotion**: Extend to handle multiple leg failures
4. **Terrain Adaptation**: Specialized strategies for different terrain types
5. **Energy Optimization**: Focus on energy-efficient three-legged locomotion patterns
