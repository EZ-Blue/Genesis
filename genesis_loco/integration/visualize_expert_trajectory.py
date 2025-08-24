#!/usr/bin/env python3
"""
Expert Trajectory Visualization Script

Loads LocoMujoco walking data through the data bridge and visualizes the full trajectory
in Genesis to help identify individual walking cycles for single-trajectory training.

This script will:
1. Load the full walking trajectory from LocoMujoco
2. Play it in Genesis with visualization
3. Output trajectory analysis (timesteps, cycles, etc.)
4. Help identify good segments for single-trajectory behavior cloning
"""

import torch
import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt

sys.path.append('/home/ez/Documents/Genesis/genesis_loco')

import genesis as gs
from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.data_bridge import LocoMujocoDataBridge


def safe_init_genesis():
    """Safely initialize Genesis"""
    try:
        gs.init(backend=gs.gpu)
        return True
    except Exception as e:
        if "already initialized" in str(e):
            return True
        else:
            print(f"Genesis initialization failed: {e}")
            return False


class TrajectoryVisualizer:
    """Visualize and analyze expert walking trajectory"""
    
    def __init__(self):
        print("🎬 Expert Trajectory Visualizer")
        print("=" * 60)
        
        # Initialize Genesis
        if not safe_init_genesis():
            raise RuntimeError("Failed to initialize Genesis")
        print("✅ Genesis initialized")
        
        # Create environment
        self.env = SkeletonHumanoidEnv(
            num_envs=1,
            episode_length_s=60.0,  # Long episode for full trajectory
            dt=0.01,
            show_viewer=True,
            use_box_feet=True
        )
        print(f"✅ Environment created: {self.env.num_observations} obs, {self.env.num_actions} actions")
        
        # Create data bridge
        self.data_bridge = LocoMujocoDataBridge(self.env)
        success = self.data_bridge.load_trajectory("walk")
        if not success:
            raise RuntimeError("Failed to load walking trajectory")
        
        self.trajectory_length = self.data_bridge.trajectory_length
        self.trajectory_freq = self.data_bridge.trajectory_frequency
        
        print(f"✅ Trajectory loaded:")
        print(f"   Total timesteps: {self.trajectory_length}")
        print(f"   Frequency: {self.trajectory_freq} Hz")
        print(f"   Duration: {self.trajectory_length / self.trajectory_freq:.2f} seconds")
        
        # Reset environment
        self.env.reset()
        
    def analyze_trajectory_structure(self):
        """Analyze trajectory to identify walking cycles"""
        print(f"\n🔍 ANALYZING TRAJECTORY STRUCTURE")
        print("=" * 60)
        
        # Sample trajectory data for analysis
        sample_interval = max(1, self.trajectory_length // 1000)  # Sample ~1000 points
        timesteps = np.arange(0, self.trajectory_length, sample_interval)
        
        # Extract key features for cycle detection
        root_positions = []
        root_heights = []
        left_foot_heights = []
        right_foot_heights = []
        
        print(f"📊 Sampling trajectory every {sample_interval} timesteps...")
        
        for i, timestep in enumerate(timesteps):
            if i % 100 == 0:
                progress = (i / len(timesteps)) * 100
                print(f"   Progress: {progress:.1f}%")
            
            state = self.data_bridge.get_trajectory_state(timestep)
            if state is None:
                continue
                
            # Extract root position and orientation
            root_pos = state['root_pos']
            if hasattr(root_pos, 'cpu'):
                root_pos = root_pos.cpu().numpy()
            
            root_positions.append(root_pos[0])  # Forward position
            root_heights.append(root_pos[2])    # Height
            
            # Apply state to get foot positions
            try:
                self.data_bridge.apply_trajectory_state(state, torch.tensor([0], device=self.env.device))
                
                # Get foot heights (approximate from robot state)
                # This is a rough estimate - you might need to adjust based on your skeleton
                if hasattr(self.env, 'robot'):
                    # Try to get foot positions - this might need adjustment based on your skeleton structure
                    left_foot_heights.append(root_pos[2] - 0.5)  # Placeholder
                    right_foot_heights.append(root_pos[2] - 0.5)  # Placeholder
                else:
                    left_foot_heights.append(0.0)
                    right_foot_heights.append(0.0)
                    
            except Exception as e:
                left_foot_heights.append(0.0)
                right_foot_heights.append(0.0)
        
        # Convert to numpy arrays
        timesteps_sampled = timesteps[:len(root_positions)]
        root_positions = np.array(root_positions)
        root_heights = np.array(root_heights)
        
        print(f"✅ Analysis complete: {len(root_positions)} samples analyzed")
        
        # Basic statistics
        print(f"\n📊 TRAJECTORY STATISTICS:")
        print(f"   Forward distance: {root_positions[-1] - root_positions[0]:.3f}m")
        print(f"   Average speed: {(root_positions[-1] - root_positions[0]) / (self.trajectory_length / self.trajectory_freq):.3f} m/s")
        print(f"   Height range: [{root_heights.min():.3f}, {root_heights.max():.3f}]m")
        print(f"   Height variation: {root_heights.std():.6f}m")
        
        # Estimate walking cycles using forward velocity
        forward_velocity = np.diff(root_positions)
        
        # Simple cycle detection: look for patterns in forward velocity
        # This is a rough estimate - walking cycles typically show periodic patterns
        if len(forward_velocity) > 10:
            # Look for velocity variations that might indicate steps
            velocity_std = np.std(forward_velocity)
            velocity_mean = np.mean(forward_velocity)
            
            print(f"\n🚶 WALKING PATTERN ANALYSIS:")
            print(f"   Average forward velocity: {velocity_mean:.6f} m/timestep")
            print(f"   Velocity variation (std): {velocity_std:.6f}")
            
            # Estimate cycle length (very rough)
            # Typical human walking: ~1.2-1.4 steps/second, so ~0.7-0.8 seconds per step
            # Full gait cycle = 2 steps = ~1.4-1.6 seconds
            estimated_cycle_duration = 1.5  # seconds
            estimated_cycle_timesteps = int(estimated_cycle_duration * self.trajectory_freq)
            
            print(f"   Estimated cycle duration: {estimated_cycle_duration}s")
            print(f"   Estimated cycle length: {estimated_cycle_timesteps} timesteps")
            print(f"   Estimated number of cycles: {self.trajectory_length // estimated_cycle_timesteps}")
        
        # Save analysis data
        analysis_data = {
            'timesteps': timesteps_sampled,
            'root_positions': root_positions,
            'root_heights': root_heights,
            'forward_velocity': forward_velocity,
            'trajectory_length': self.trajectory_length,
            'trajectory_freq': self.trajectory_freq
        }
        
        return analysis_data
    
    def suggest_single_trajectory_segments(self, analysis_data):
        """Suggest good segments for single-trajectory training"""
        print(f"\n💡 SINGLE TRAJECTORY SEGMENT SUGGESTIONS")
        print("=" * 60)
        
        # Parameters for segment selection
        min_segment_length = int(2.0 * self.trajectory_freq)  # 2 seconds minimum
        max_segment_length = int(5.0 * self.trajectory_freq)  # 5 seconds maximum
        overlap_buffer = int(0.5 * self.trajectory_freq)     # 0.5 second buffer
        
        print(f"Segment criteria:")
        print(f"   Minimum length: {min_segment_length} timesteps ({min_segment_length/self.trajectory_freq:.1f}s)")
        print(f"   Maximum length: {max_segment_length} timesteps ({max_segment_length/self.trajectory_freq:.1f}s)")
        print(f"   Buffer between segments: {overlap_buffer} timesteps")
        
        # Generate suggested segments
        suggested_segments = []
        
        # Strategy 1: Regular intervals across trajectory
        num_segments = 5  # Number of segments to suggest
        segment_starts = np.linspace(
            overlap_buffer, 
            self.trajectory_length - max_segment_length - overlap_buffer, 
            num_segments, 
            dtype=int
        )
        
        for i, start in enumerate(segment_starts):
            end = start + max_segment_length
            duration = (end - start) / self.trajectory_freq
            
            # Calculate approximate forward distance for this segment
            start_state = self.data_bridge.get_trajectory_state(start)
            end_state = self.data_bridge.get_trajectory_state(end)
            
            if start_state and end_state:
                start_pos = start_state['root_pos']
                end_pos = end_state['root_pos']
                
                if hasattr(start_pos, 'cpu'):
                    start_pos = start_pos.cpu().numpy()
                    end_pos = end_pos.cpu().numpy()
                
                distance = end_pos[0] - start_pos[0]
                avg_speed = distance / duration
                
                segment_info = {
                    'id': i + 1,
                    'start_timestep': start,
                    'end_timestep': end,
                    'duration': duration,
                    'distance': distance,
                    'avg_speed': avg_speed,
                    'start_time': start / self.trajectory_freq,
                    'end_time': end / self.trajectory_freq
                }
                
                suggested_segments.append(segment_info)
        
        # Display suggestions
        print(f"\n📋 SUGGESTED SEGMENTS FOR SINGLE-TRAJECTORY TRAINING:")
        print(f"{'ID':<3} {'Start':<8} {'End':<8} {'Duration':<8} {'Distance':<8} {'Speed':<8} {'Time Range'}")
        print("-" * 70)
        
        for seg in suggested_segments:
            print(f"{seg['id']:<3} {seg['start_timestep']:<8} {seg['end_timestep']:<8} "
                  f"{seg['duration']:.1f}s{'':<3} {seg['distance']:.2f}m{'':<3} "
                  f"{seg['avg_speed']:.2f}m/s{'':<2} {seg['start_time']:.1f}-{seg['end_time']:.1f}s")
        
        return suggested_segments
    
    def visualize_trajectory(self, start_timestep=0, end_timestep=None, playback_speed=1.0):
        """Visualize trajectory in Genesis"""
        if end_timestep is None:
            end_timestep = self.trajectory_length
        
        end_timestep = min(end_timestep, self.trajectory_length)
        total_steps = end_timestep - start_timestep
        
        print(f"\n🎬 VISUALIZING TRAJECTORY")
        print("=" * 60)
        print(f"Range: timesteps {start_timestep} to {end_timestep}")
        print(f"Duration: {total_steps / self.trajectory_freq:.2f} seconds")
        print(f"Playback speed: {playback_speed}x")
        print(f"Press Ctrl+C to stop early")
        
        env_ids = torch.tensor([0], device=self.env.device)
        
        try:
            for step in range(total_steps):
                timestep = start_timestep + step
                
                # Get and apply trajectory state
                state = self.data_bridge.get_trajectory_state(timestep)
                if state is not None:
                    self.data_bridge.apply_trajectory_state(state, env_ids)
                
                # Progress updates
                if step % 100 == 0:
                    progress = (step / total_steps) * 100
                    current_time = timestep / self.trajectory_freq
                    print(f"   Step {step:4d}/{total_steps} ({progress:5.1f}%) - Time: {current_time:.2f}s")
                
                # Playback speed control
                if playback_speed < 1.0:
                    time.sleep((1.0 - playback_speed) * 0.01)
        
        except KeyboardInterrupt:
            print(f"\n🛑 Visualization stopped by user at step {step}")
        
        print(f"✅ Visualization complete!")
    
    def plot_trajectory_analysis(self, analysis_data, save_path="trajectory_analysis.png"):
        """Plot trajectory analysis"""
        print(f"\n📊 PLOTTING TRAJECTORY ANALYSIS")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Forward position over time
        time_axis = analysis_data['timesteps'] / self.trajectory_freq
        
        axes[0, 0].plot(time_axis, analysis_data['root_positions'])
        axes[0, 0].set_title('Forward Position Over Time')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Forward Position (m)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Height over time
        axes[0, 1].plot(time_axis, analysis_data['root_heights'])
        axes[0, 1].set_title('Root Height Over Time')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Height (m)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Forward velocity
        if len(analysis_data['forward_velocity']) > 0:
            velocity_time = time_axis[1:]  # One less point due to diff
            axes[1, 0].plot(velocity_time, analysis_data['forward_velocity'])
            axes[1, 0].set_title('Forward Velocity')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Velocity (m/timestep)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Trajectory overview
        axes[1, 1].text(0.1, 0.7, f"Trajectory Length: {self.trajectory_length} timesteps", fontsize=10)
        axes[1, 1].text(0.1, 0.6, f"Duration: {self.trajectory_length/self.trajectory_freq:.1f} seconds", fontsize=10)
        axes[1, 1].text(0.1, 0.5, f"Frequency: {self.trajectory_freq} Hz", fontsize=10)
        axes[1, 1].text(0.1, 0.4, f"Total Distance: {analysis_data['root_positions'][-1] - analysis_data['root_positions'][0]:.2f}m", fontsize=10)
        axes[1, 1].text(0.1, 0.3, f"Avg Speed: {(analysis_data['root_positions'][-1] - analysis_data['root_positions'][0]) / (self.trajectory_length/self.trajectory_freq):.2f} m/s", fontsize=10)
        axes[1, 1].set_title('Trajectory Summary')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Analysis plot saved to: {save_path}")
        plt.show()


def main():
    """Main function"""
    print("🎬 EXPERT TRAJECTORY VISUALIZATION")
    print("=" * 60)
    print("This script helps you:")
    print("1. Visualize the full LocoMujoco walking trajectory")
    print("2. Analyze trajectory structure and patterns")
    print("3. Identify good segments for single-trajectory training")
    
    try:
        # Create visualizer
        visualizer = TrajectoryVisualizer()
        
        # Analyze trajectory
        analysis_data = visualizer.analyze_trajectory_structure()
        
        # Get segment suggestions
        segments = visualizer.suggest_single_trajectory_segments(analysis_data)
        
        # Plot analysis
        visualizer.plot_trajectory_analysis(analysis_data)
        
        # Interactive options
        print(f"\n🎮 VISUALIZATION OPTIONS:")
        print("1. Play full trajectory")
        print("2. Play specific segment")
        print("3. Play at different speed")
        print("4. Skip visualization")
        
        choice = input("Select option (1/2/3/4): ").strip()
        
        if choice == "1":
            print("Playing full trajectory...")
            visualizer.visualize_trajectory()
            
        elif choice == "2":
            print("Available segments:")
            for seg in segments:
                print(f"  {seg['id']}: timesteps {seg['start_timestep']}-{seg['end_timestep']} ({seg['duration']:.1f}s)")
            
            try:
                seg_id = int(input("Enter segment ID to visualize: "))
                if 1 <= seg_id <= len(segments):
                    seg = segments[seg_id - 1]
                    print(f"Playing segment {seg_id}...")
                    visualizer.visualize_trajectory(seg['start_timestep'], seg['end_timestep'])
                else:
                    print("Invalid segment ID")
            except ValueError:
                print("Invalid input")
                
        elif choice == "3":
            try:
                speed = float(input("Enter playback speed (0.1-2.0): "))
                speed = max(0.1, min(2.0, speed))
                print(f"Playing at {speed}x speed...")
                visualizer.visualize_trajectory(playback_speed=speed)
            except ValueError:
                print("Invalid speed, using normal speed")
                visualizer.visualize_trajectory()
        
        print(f"\n✅ Trajectory analysis complete!")
        print(f"💡 Use the suggested segments for single-trajectory behavior cloning")
        
    except Exception as e:
        print(f"\n❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())