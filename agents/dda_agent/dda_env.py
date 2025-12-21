"""
DDA Environment Wrapper

Wraps the game environment to provide DDA agent's perspective.
Tracks metrics, extracts state, applies difficulty adjustments, and computes rewards.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .state_extractor import DDAStateExtractor
from .difficulty_manager import DifficultyManager


class DDAEnvironment:
    """
    Environment wrapper for DDA agent.
    
    The DDA agent acts at fixed intervals (e.g., every 5 seconds) and adjusts
    difficulty parameters to maintain balanced challenge.
    """
    
    def __init__(self, game_env, base_params: Dict, action_interval: int = 300, fps: int = 60):
        """
        Initialize DDA environment.
        
        Args:
            game_env: The underlying game environment (ZombieShooterEnv)
            base_params: Base difficulty parameters
            action_interval: Frames between DDA actions (default 300 = 5 seconds at 60 FPS)
            fps: Frames per second
        """
        self.game_env = game_env
        self.base_params = base_params
        self.action_interval = action_interval
        self.fps = fps
        
        # Initialize components
        self.state_extractor = DDAStateExtractor(window_size=600, fps=fps)  # 10 second window
        self.difficulty_manager = DifficultyManager(base_params)
        
        # Player agent (set externally)
        self.player_agent = None
        
        # Episode tracking
        self.episode_steps = 0
        self.dda_action_steps = 0  # Steps since last DDA action
        self.episode_reward = 0.0
        self.episode_length = 0
        
        # Metrics tracking
        self.last_player_health = 100.0
        self.player_alive = True
        self.total_damage_taken = 0.0
        self.total_damage_dealt = 0.0
        self.total_kills = 0
        
        # Track previous state for metrics
        self.prev_zombies_killed = 0
        self.prev_bullets_fired = 0
        self.prev_bullets_hit = 0
        
        # Target "flow zone" parameters
        self.target_health_min = 0.3
        self.target_health_max = 0.7
        self.target_zombies_min = 3
        self.target_zombies_max = 10
        
    def reset(self) -> np.ndarray:
        """
        Reset environment and return initial DDA state.
        
        Returns:
            Initial DDA state vector
        """
        # Reset game environment
        game_state = self.game_env.reset()
        
        # Reset DDA components
        self.state_extractor.reset()
        self.difficulty_manager.reset()
        
        # Apply base parameters to game
        self._apply_difficulty_params()
        
        # Reset tracking
        self.episode_steps = 0
        self.dda_action_steps = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.last_player_health = 100.0
        self.player_alive = True
        self.total_damage_taken = 0.0
        self.total_damage_dealt = 0.0
        self.total_kills = 0
        
        # Extract initial DDA state
        dda_state = self._extract_dda_state()
        
        return dda_state
    
    def step(self, dda_action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one DDA step (may span multiple game steps).
        
        The DDA agent acts at fixed intervals. Between actions, we run the game
        with the current difficulty settings.
        
        Args:
            dda_action: DDA action (0-4)
        
        Returns:
            observation: DDA state
            reward: DDA reward for maintaining balanced challenge
            done: Whether episode is finished
            info: Additional information
        """
        # Apply DDA action if it's time
        if self.dda_action_steps >= self.action_interval:
            # Apply difficulty adjustment
            self.difficulty_manager.apply_action(dda_action)
            self._apply_difficulty_params()
            self.dda_action_steps = 0
        
        # Run game for action_interval frames (or until episode ends)
        game_reward = 0.0
        done = False
        info = {}
        
        steps_to_run = min(self.action_interval - self.dda_action_steps, 100)  # Run up to 100 steps at a time
        
        for _ in range(steps_to_run):
            # Get action from player agent (if available)
            # For now, we'll use a simple heuristic or let the game run
            # In training, we'll use the trained player agent
            
            # Run one game step
            # Note: We need to get the player action somehow
            # For now, assume we have access to player agent or use random
            player_action = self._get_player_action()
            
            game_state, step_reward, game_done, game_info = self.game_env.step(player_action)
            game_reward += step_reward
            
            # Extract metrics from game info and environment
            metrics_info = self._extract_game_metrics(game_info)
            
            # Update metrics tracking
            self._update_metrics(metrics_info)
            
            # Check if episode ended
            if game_done:
                done = True
                info.update(game_info)
                break
            
            self.episode_steps += 1
            self.dda_action_steps += 1
        
        # Extract DDA state
        dda_state = self._extract_dda_state()
        
        # Compute DDA reward
        dda_reward = self._compute_dda_reward(done)
        self.episode_reward += dda_reward
        
        # Update info
        info['dda_reward'] = dda_reward
        info['episode_reward'] = self.episode_reward
        info['difficulty_params'] = self.difficulty_manager.get_current_params()
        info['player_health'] = self.last_player_health
        # Get zombies count directly from game environment
        if hasattr(self.game_env, 'zombies'):
            info['zombies_alive'] = len(self.game_env.zombies)
        else:
            info['zombies_alive'] = 0
        
        return dda_state, dda_reward, done, info
    
    def _get_player_action(self) -> int:
        """
        Get action from player agent.
        
        In training, this will use the trained player agent.
        For now, returns a placeholder.
        """
        if hasattr(self, 'player_agent') and self.player_agent is not None:
            # Get current game state
            game_state = self.game_env._get_state()
            # Get action from player agent
            action, _, _ = self.player_agent.select_action(game_state, deterministic=False)
            return action
        # Default to idle if no player agent
        return 4  # Idle action
    
    def set_player_agent(self, player_agent):
        """Set the player agent to use for game actions."""
        self.player_agent = player_agent
    
    def _extract_game_metrics(self, game_info: Dict) -> Dict:
        """
        Extract metrics from game environment and info.
        
        Returns:
            Dictionary with extracted metrics
        """
        # Get player health
        if hasattr(self.game_env, 'player') and isinstance(self.game_env.player, dict):
            player_health = self.game_env.player.get('health', 100.0)
        else:
            player_health = game_info.get('player_health', self.last_player_health)
        
        # Calculate damage taken
        damage_taken = max(0, self.last_player_health - player_health)
        
        # Track kills
        current_kills = game_info.get('zombies_killed', 0)
        zombie_killed = (current_kills > self.prev_zombies_killed)
        if zombie_killed:
            self.prev_zombies_killed = current_kills
        
        # Track shooting (simplified: check if bullets were fired)
        # We can infer from cooldown or bullet count
        shot_fired = False
        shot_hit = False
        if hasattr(self.game_env, 'player') and isinstance(self.game_env.player, dict):
            # If cooldown just started, a shot was fired
            shoot_cooldown = self.game_env.player.get('shoot_cooldown', 0)
            if shoot_cooldown > 0:
                shot_fired = True
            # If a zombie was killed, a shot likely hit
            if zombie_killed:
                shot_hit = True
        
        # Track pickups (check if player health increased or machinegun acquired)
        pickup_collected = False
        if hasattr(self.game_env, 'player') and isinstance(self.game_env.player, dict):
            if player_health > self.last_player_health:
                pickup_collected = True
            # Check machinegun pickup
            has_mg = self.game_env.player.get('has_machinegun', False)
            if has_mg and not hasattr(self, '_had_machinegun'):
                pickup_collected = True
                self._had_machinegun = True
            elif not has_mg:
                self._had_machinegun = False
        
        # Damage dealt (simplified: assume damage when zombie killed)
        damage_dealt = 0
        if zombie_killed:
            # Estimate damage (zombie health)
            damage_dealt = 30  # Average zombie health
        
        return {
            'player_health': player_health,
            'damage_taken': damage_taken,
            'damage_dealt': damage_dealt,
            'zombie_killed': zombie_killed,
            'shot_fired': shot_fired,
            'shot_hit': shot_hit,
            'pickup_collected': pickup_collected,
        }
    
    def _update_metrics(self, metrics_info: Dict):
        """Update metrics from game step."""
        # Get player health from game environment
        if hasattr(self.game_env, 'player'):
            player_health = self.game_env.player.get('health', self.last_player_health)
        else:
            player_health = game_info.get('player_health', self.last_player_health)
        
        # Calculate damage taken this step
        damage_taken = max(0, self.last_player_health - player_health)
        self.last_player_health = player_health
        
        # Update state extractor with metrics_info
        self.state_extractor.update(metrics_info)
        
        # Extract values for tracking
        damage_taken = metrics_info.get('damage_taken', 0)
        zombie_killed = metrics_info.get('zombie_killed', False)
        damage_dealt = metrics_info.get('damage_dealt', 0)
        player_health = metrics_info.get('player_health', self.last_player_health)
        
        # Track totals
        if damage_taken > 0:
            self.total_damage_taken += damage_taken
        if zombie_killed:
            self.total_kills += 1
        if damage_dealt > 0:
            self.total_damage_dealt += damage_dealt
        
        # Check if player died
        if player_health <= 0:
            self.player_alive = False
    
    def _extract_dda_state(self) -> np.ndarray:
        """Extract DDA state from current game state."""
        # Get game state info
        zombies_alive = 0
        machinegun_ammo = 0
        if hasattr(self.game_env, 'zombies'):
            zombies_alive = len(self.game_env.zombies)
        if hasattr(self.game_env, 'player') and isinstance(self.game_env.player, dict):
            machinegun_ammo = self.game_env.player.get('machinegun_ammo', 0)
        
        game_state_info = {
            'player_health': self.last_player_health,
            'zombies_alive': zombies_alive,
            'max_zombies': 20,
            'machinegun_ammo': machinegun_ammo,
            'max_ammo': 100,
        }
        
        # Extract state
        dda_state = self.state_extractor.extract_state(game_state_info)
        
        return dda_state
    
    def _compute_dda_reward(self, done: bool) -> float:
        """
        Compute DDA reward for maintaining balanced challenge.
        
        Reward principles:
        - Surviving is good, dying is bad
        - Extreme ease is bad (player always full health)
        - Extreme difficulty is bad (player overwhelmed)
        - Encourage mid-range player health and pressure
        
        Returns:
            DDA reward
        """
        reward = 0.0
        
        # 1. Survival reward
        if self.player_alive:
            reward += 1.0
        else:
            reward -= 5.0  # Death penalty
            return reward
        
        # 2. Health balance reward (target: 0.3 to 0.7)
        health_ratio = self.last_player_health / 100.0
        if self.target_health_min <= health_ratio <= self.target_health_max:
            # In target zone: bonus
            reward += 2.0
        else:
            # Outside target zone: penalty proportional to distance
            if health_ratio < self.target_health_min:
                # Too low (too hard)
                distance = self.target_health_min - health_ratio
                reward -= distance * 3.0
            else:
                # Too high (too easy)
                distance = health_ratio - self.target_health_max
                reward -= distance * 2.0
        
        # 3. Zombie count balance (target: 3-10 zombies)
        zombies_alive = 0
        if hasattr(self.game_env, 'zombies'):
            zombies_alive = len(self.game_env.zombies)
        if self.target_zombies_min <= zombies_alive <= self.target_zombies_max:
            reward += 0.5
        else:
            if zombies_alive < self.target_zombies_min:
                # Too few zombies (too easy)
                reward -= 0.5
            else:
                # Too many zombies (too hard)
                reward -= 1.0
        
        # 4. Damage taken penalty (if too high, difficulty is too high)
        avg_damage_taken = self.state_extractor.damage_taken_history
        if len(avg_damage_taken) > 0:
            recent_damage = sum(list(avg_damage_taken)[-60:])  # Last second
            if recent_damage > 30:  # Too much damage
                reward -= 1.0
        
        # 5. Kill rate balance (encourage active engagement)
        kill_rate = self.state_extractor.kills_history
        if len(kill_rate) > 0:
            recent_kills = sum(list(kill_rate)[-300:])  # Last 5 seconds
            if 1 <= recent_kills <= 5:  # Good kill rate
                reward += 0.5
            elif recent_kills == 0:  # No kills (too easy or too hard)
                reward -= 0.3
        
        return reward
    
    def _apply_difficulty_params(self):
        """Apply current difficulty parameters to game environment."""
        params = self.difficulty_manager.get_current_params()
        
        # Update game environment parameters
        if hasattr(self.game_env, 'ZOMBIE_SPEED'):
            self.game_env.ZOMBIE_SPEED = params['ZOMBIE_SPEED']
        if hasattr(self.game_env, 'SPAWN_RATE'):
            self.game_env.SPAWN_RATE = int(params['SPAWN_RATE'])
            self.game_env.spawn_rate = int(params['SPAWN_RATE'])
        if hasattr(self.game_env, 'ZOMBIE_ATTACK_COOLDOWN'):
            self.game_env.ZOMBIE_ATTACK_COOLDOWN = int(params['ZOMBIE_ATTACK_COOLDOWN'])
        
        if hasattr(self.game_env, 'ZOMBIE_NORMAL_HEALTH'):
            self.game_env.ZOMBIE_NORMAL_HEALTH = int(params['ZOMBIE_NORMAL_HEALTH'])
        if hasattr(self.game_env, 'ZOMBIE_STRONG_HEALTH'):
            self.game_env.ZOMBIE_STRONG_HEALTH = int(params['ZOMBIE_STRONG_HEALTH'])
        
        if hasattr(self.game_env, 'PLAYER_SPEED'):
            self.game_env.PLAYER_SPEED = params['PLAYER_SPEED']
        if hasattr(self.game_env, 'PISTOL_DAMAGE'):
            self.game_env.PISTOL_DAMAGE = int(params['PISTOL_DAMAGE'])
        if hasattr(self.game_env, 'MACHINEGUN_DAMAGE'):
            self.game_env.MACHINEGUN_DAMAGE = int(params['MACHINEGUN_DAMAGE'])
        
        if hasattr(self.game_env, 'HEALTH_PICKUP_DROP_PROBABILITY'):
            self.game_env.HEALTH_PICKUP_DROP_PROBABILITY = params['HEALTH_PICKUP_DROP_PROBABILITY']
        if hasattr(self.game_env, 'MACHINEGUN_PICKUP_DROP_PROBABILITY'):
            self.game_env.MACHINEGUN_PICKUP_DROP_PROBABILITY = params['MACHINEGUN_PICKUP_DROP_PROBABILITY']
        
        if hasattr(self.game_env, 'PISTOL_COOLDOWN'):
            self.game_env.PISTOL_COOLDOWN = int(params['PISTOL_COOLDOWN'])
        if hasattr(self.game_env, 'MACHINEGUN_COOLDOWN'):
            self.game_env.MACHINEGUN_COOLDOWN = int(params['MACHINEGUN_COOLDOWN'])

