import pygame
import random
import math
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)

class PollenType(Enum):
    HIGH_QUALITY = 0
    MEDIUM_QUALITY = 1
    LOW_QUALITY = 2

@dataclass
class PollenSource:
    x: float
    y: float
    pollen_type: PollenType
    reward: float
    color: Tuple[int, int, int]
    radius: int = 15
    
    def get_reward(self):
        return self.reward

class Bee:
    def __init__(self, x, y, bee_id):
        self.x = x
        self.y = y
        self.bee_id = bee_id
        self.target = None
        self.speed = 2 + random.uniform(-0.5, 0.5)
        self.fitness = 0.0
        self.lifetime_reward = 0.0
        
        # Preference weights for different pollen sources (sum to 1)
        self.preferences = np.random.dirichlet([1, 1, 1])  # Random initial preferences
        
        # Movement 
        self.angle = random.uniform(0, 2 * math.pi)
        self.last_decision_time = 0
        self.decision_interval = random.randint(60, 180)  # Frames between decisions
        
    def update_preferences_replicator_dynamics(self, payoff_matrix, population_preferences, dt=0.01):
        """Update preferences using replicator dynamics"""
        avg_payoffs = np.dot(payoff_matrix, population_preferences)
        avg_fitness = np.dot(self.preferences, avg_payoffs)
        
        for i in range(len(self.preferences)):
            self.preferences[i] += dt * self.preferences[i] * (avg_payoffs[i] - avg_fitness)
        
        # Normalize to ensure preferences sum to 1
        self.preferences = np.maximum(self.preferences, 0.001)  # Prevent negative values
        self.preferences /= np.sum(self.preferences)
    
    def mutate(self, mutation_rate=0.1):
        """Apply mutation to preferences"""
        if random.random() < mutation_rate:
            # Add small random noise to preferences
            noise = np.random.normal(0, 0.1, len(self.preferences))
            self.preferences += noise
            self.preferences = np.maximum(self.preferences, 0.001)
            self.preferences /= np.sum(self.preferences)
    
    def choose_pollen_source(self, pollen_sources):
        """Choose pollen source based on preferences and distance"""
        if not pollen_sources:
            return None
        
        best_source = None
        best_score = -float('inf')
        
        for source in pollen_sources:
            # Distance factor (closer is better)
            distance = math.sqrt((self.x - source.x)**2 + (self.y - source.y)**2)
            distance_score = 1.0 / (1.0 + distance / 100.0)
            
            # Preference factor
            preference_score = self.preferences[source.pollen_type.value]
            
            # Combined score
            total_score = preference_score * distance_score
            
            if total_score > best_score:
                best_score = total_score
                best_source = source
        
        return best_source
    
    def move_towards_target(self, target):
        """Move towards the target pollen source"""
        if target:
            dx = target.x - self.x
            dy = target.y - self.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance > 5:  # Not at target yet
                self.x += (dx / distance) * self.speed
                self.y += (dy / distance) * self.speed
                return False
            else:
                # Reached target
                reward = target.get_reward()
                self.fitness += reward
                self.lifetime_reward += reward
                return True
        return False
    
    def random_walk(self):
        """Perform random walk when no target"""
        self.angle += random.uniform(-0.3, 0.3)
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        
        # Keep within bounds
        self.x = max(10, min(SCREEN_WIDTH - 10, self.x))
        self.y = max(10, min(SCREEN_HEIGHT - 10, self.y))
    
    def update(self, pollen_sources, frame_count):
        """Update bee behavior"""
        reached_target = False
        
        if self.target:
            reached_target = self.move_towards_target(self.target)
            if reached_target:
                self.target = None
        
        # Make decisions at intervals or after reaching target
        if (frame_count - self.last_decision_time > self.decision_interval) or reached_target:
            self.target = self.choose_pollen_source(pollen_sources)
            self.last_decision_time = frame_count
            self.decision_interval = random.randint(60, 180)
        
        if not self.target:
            self.random_walk()
    
    def draw(self, screen):
        """Draw the bee with optimized rendering"""
        # Simple circle for better performance
        pygame.draw.circle(screen, YELLOW, (int(self.x), int(self.y)), 2)
        # Always draw direction indicator (simplified)
        end_x = self.x + math.cos(self.angle) * 6
        end_y = self.y + math.sin(self.angle) * 6
        pygame.draw.line(screen, BLACK, (self.x, self.y), (end_x, end_y), 1)

class BeeSwarmSimulation:
    def __init__(self, num_bees=50):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Bee Evolution Simulation (Max Population: 5,000)")
        self.clock = pygame.time.Clock()
        
        # Initialize pollen sources
        self.pollen_sources = [
            PollenSource(200, 200, PollenType.HIGH_QUALITY, 10.0, GREEN),
            PollenSource(600, 150, PollenType.HIGH_QUALITY, 10.0, GREEN),
            PollenSource(1000, 300, PollenType.HIGH_QUALITY, 10.0, GREEN),
            
            PollenSource(300, 500, PollenType.MEDIUM_QUALITY, 5.0, ORANGE),
            PollenSource(700, 400, PollenType.MEDIUM_QUALITY, 5.0, ORANGE),
            PollenSource(900, 600, PollenType.MEDIUM_QUALITY, 5.0, ORANGE),
            
            PollenSource(150, 700, PollenType.LOW_QUALITY, 1.0, RED),
            PollenSource(500, 650, PollenType.LOW_QUALITY, 1.0, RED),
            PollenSource(800, 100, PollenType.LOW_QUALITY, 1.0, RED),
            PollenSource(400, 300, PollenType.LOW_QUALITY, 1.0, RED),
        ]
        
        # Initialize bees
        self.bees = []
        for i in range(num_bees):
            x = random.uniform(50, SCREEN_WIDTH - 50)
            y = random.uniform(50, SCREEN_HEIGHT - 50)
            self.bees.append(Bee(x, y, i))
        
        # Evolution parameters
        self.generation = 0
        self.frame_count = 0
        self.generation_length = 1800  # 30 seconds at 60 FPS
        self.mutation_rate = 0.05
        self.manual_evolution = True  # Start in manual mode
        
        # POPULATION PARAMETERS - SET TO 5000 MAX
        self.initial_population = num_bees
        self.population_growth_rate = 0.3  # 30% growth per generation
        self.max_population = 5000  # HARD CODED TO 5000 MAXIMUM
        
        # Payoff matrix for replicator dynamics
        self.payoff_matrix = np.array([
            [10.0, 8.0, 2.0],   # High quality pollen interactions
            [8.0, 5.0, 3.0],    # Medium quality pollen interactions
            [2.0, 3.0, 1.0]     # Low quality pollen interactions
        ])
        
        # Statistics
        self.generation_stats = []
        self.population_preferences_history = []
        
        # Font for text
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Print initialization info
        print(" BEE SWARM EVOLUTION INITIALIZED ")
        print(f"Starting Population: {len(self.bees)} bees")
        print(f"MAXIMUM POPULATION: {self.max_population} bees")
        print(f"Growth Rate: {self.population_growth_rate*100:.0f}% per generation")
        print("=" * 50)
    
    def add_new_bees(self, num_new_bees):
        """Add new bees to the population through reproduction"""
        if len(self.bees) >= self.max_population:
            print(f"Cannot add bees - at maximum population of {self.max_population}")
            return 0
        
        # Limit by max population of 5000
        num_new_bees = min(num_new_bees, self.max_population - len(self.bees))
        
        added_bees = 0
        for _ in range(num_new_bees):
            # Select parent based on lifetime reward (fitness)
            if not self.bees:
                break
                
            total_fitness = sum(max(bee.lifetime_reward, 0.1) for bee in self.bees)
            if total_fitness <= 0:
                parent = random.choice(self.bees)
            else:
                # Fitness-proportional selection
                r = random.uniform(0, total_fitness)
                cumulative = 0
                parent = None
                for bee in self.bees:
                    cumulative += max(bee.lifetime_reward, 0.1)
                    if cumulative >= r:
                        parent = bee
                        break
                
                if not parent:
                    parent = random.choice(self.bees)
            
            # Create offspring near parent's location
            x = parent.x + random.uniform(-50, 50)
            y = parent.y + random.uniform(-50, 50)
            x = max(50, min(SCREEN_WIDTH - 50, x))
            y = max(50, min(SCREEN_HEIGHT - 50, y))
            
            new_bee = Bee(x, y, len(self.bees))
            new_bee.preferences = parent.preferences.copy()
            new_bee.mutate(self.mutation_rate)
            self.bees.append(new_bee)
            added_bees += 1
            
            # Safety check to never exceed 5000
            if len(self.bees) >= 5000:
                print("SAFETY: Reached 5000 bee limit")
                break
        
        return added_bees
    
    def moran_process_selection(self):
        """Implement Moran process for evolution"""
        if len(self.bees) < 2:
            return
        
        # Calculate fitness-proportional selection probabilities
        total_fitness = sum(max(bee.fitness, 0.1) for bee in self.bees)
        if total_fitness == 0:
            return
        
        # Select parent proportional to fitness
        r = random.uniform(0, total_fitness)
        cumulative = 0
        parent = None
        for bee in self.bees:
            cumulative += max(bee.fitness, 0.1)
            if cumulative >= r:
                parent = bee
                break
        
        if not parent:
            parent = random.choice(self.bees)
        
        # Select random individual to replace
        replacement_idx = random.randint(0, len(self.bees) - 1)
        
        # Create offspring with parent's preferences
        new_bee = Bee(self.bees[replacement_idx].x, self.bees[replacement_idx].y, 
                     self.bees[replacement_idx].bee_id)
        new_bee.preferences = parent.preferences.copy()
        
        # Apply mutation
        new_bee.mutate(self.mutation_rate)
        
        # Replace the selected bee
        self.bees[replacement_idx] = new_bee
    
    def update_population_replicator_dynamics(self):
        """Update population using replicator dynamics"""
        if not self.bees:
            return
        
        # Calculate average population preferences
        avg_preferences = np.mean([bee.preferences for bee in self.bees], axis=0)
        
        # Update each bee's preferences
        for bee in self.bees:
            bee.update_preferences_replicator_dynamics(self.payoff_matrix, avg_preferences)
    
    def evolve_population(self):
        """Apply evolutionary processes and population growth"""
        old_population = len(self.bees)
        
        # Apply multiple Moran process steps
        for _ in range(5):
            self.moran_process_selection()
        
        # Update preferences using replicator dynamics
        self.update_population_replicator_dynamics()
        
        # Population growth - LIMITED TO 5000 MAXIMUM
        current_pop = len(self.bees)
        if current_pop < 5000:  # Hard coded 5000 limit
            growth_amount = max(1, int(current_pop * self.population_growth_rate))
            added_bees = self.add_new_bees(growth_amount)
            print(f"Population growth: {old_population:,} → {len(self.bees):,} (+{added_bees:,} new bees)")
        else:
            print(f" MAXIMUM POPULATION REACHED: {len(self.bees):,}/5,000 bees")
        
        # Reset fitness for next generation (but keep lifetime_reward)
        for bee in self.bees:
            bee.fitness = 0.0
        
        self.generation += 1
    
    def collect_statistics(self):
        """Collect statistics about the current generation"""
        if not self.bees:
            return
        
        avg_preferences = np.mean([bee.preferences for bee in self.bees], axis=0)
        total_reward = sum(bee.lifetime_reward for bee in self.bees)
        avg_reward = total_reward / len(self.bees)
        
        self.generation_stats.append({
            'generation': self.generation,
            'avg_preferences': avg_preferences.copy(),
            'avg_reward': avg_reward,
            'population_size': len(self.bees)
        })
        
        self.population_preferences_history.append(avg_preferences.copy())
    
    def advance_generation_manually(self):
        """Manually advance to the next generation"""
        print(f" ADVANCING TO GENERATION {self.generation + 1} ")
        self.collect_statistics()
        self.evolve_population()
        
        # Print generation info to console
        if self.bees:
            avg_prefs = np.mean([bee.preferences for bee in self.bees], axis=0)
            pop_percent = (len(self.bees) / 5000) * 100  # Hard coded 5000
            print(f" Generation {self.generation} Summary:")
            print(f"   Population: {len(self.bees):,} / 5,000 ({pop_percent:.1f}%)")
            print(f"   Preferences → High: {avg_prefs[0]:.3f}, Med: {avg_prefs[1]:.3f}, Low: {avg_prefs[2]:.3f}")
            avg_reward = sum(bee.lifetime_reward for bee in self.bees) / len(self.bees)
            print(f"   Avg Lifetime Reward: {avg_reward:.2f}")
        print("=" * 60)
    
    def draw_ui(self):
        """Draw UI elements"""
        # Generation info
        gen_text = self.font.render(f"Generation: {self.generation}", True, BLACK)
        self.screen.blit(gen_text, (10, 10))
        
        # Population size - SHOW 5000 MAX
        pop_display = f"{len(self.bees):,} / 5,000"
        pop_text = self.font.render(f"Population: {pop_display}", True, BLACK)
        self.screen.blit(pop_text, (10, 50))
        
        # Growth rate info
        growth_text = self.small_font.render(f"Growth Rate: {self.population_growth_rate*100:.0f}% per generation", True, BLACK)
        self.screen.blit(growth_text, (10, 90))
        
        # Population percentage
        pop_percentage = (len(self.bees) / 5000) * 100  # Hard coded 5000
        percent_text = self.small_font.render(f"Progress: {pop_percentage:.1f}% of 5,000 maximum", True, BLACK)
        self.screen.blit(percent_text, (10, 115))
        
        # Simple progress bar
        bar_width = 300
        bar_height = 10
        filled_width = int((len(self.bees) / 5000) * bar_width)  # Hard coded 5000
        pygame.draw.rect(self.screen, BLACK, (10, 140, bar_width, bar_height), 2)
        pygame.draw.rect(self.screen, GREEN, (10, 140, filled_width, bar_height))
        
        # Average preferences
        if self.bees:
            avg_prefs = np.mean([bee.preferences for bee in self.bees], axis=0)
            pref_text = self.small_font.render(f"Preferences → High: {avg_prefs[0]:.3f}, Med: {avg_prefs[1]:.3f}, Low: {avg_prefs[2]:.3f}", 
                                             True, BLACK)
            self.screen.blit(pref_text, (10, 165))
        
        # Evolution mode indicator
        mode_text = "MANUAL" if self.manual_evolution else "AUTO"
        mode_surface = self.font.render(f"Mode: {mode_text}", True, BLUE)
        self.screen.blit(mode_surface, (10, 195))
        
        if self.manual_evolution:
            instruction_text = self.small_font.render("Press SPACE to advance generation", True, BLUE)
            self.screen.blit(instruction_text, (10, 235))
        else:
            time_left = (self.generation_length - (self.frame_count % self.generation_length)) // 60
            auto_text = self.small_font.render(f"Next generation in: {time_left}s", True, BLUE)
            self.screen.blit(auto_text, (10, 235))
        
        # Performance info for large populations
        if len(self.bees) > 1000:
            perf_text = self.small_font.render(f"Large pop mode: Drawing sample of bees for performance", True, PURPLE)
            self.screen.blit(perf_text, (10, 260))
        
        # Legend
        legend_y = SCREEN_HEIGHT - 120
        legend_items = [
            ("High Quality (10 pts)", GREEN),
            ("Medium Quality (5 pts)", ORANGE),
            ("Low Quality (1 pt)", RED),
            ("Bees", YELLOW)
        ]
        
        for i, (text, color) in enumerate(legend_items):
            pygame.draw.circle(self.screen, color, (30, legend_y + i * 25), 8)
            text_surface = self.small_font.render(text, True, BLACK)
            self.screen.blit(text_surface, (50, legend_y + i * 25 - 8))
    
    def run(self):
        """Main simulation loop"""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        # Reset simulation
                        print(" RESETTING SIMULATION ")
                        self.__init__(self.initial_population)
                    elif event.key == pygame.K_SPACE:
                        # Advance to next generation manually
                        if self.manual_evolution:
                            self.advance_generation_manually()
                        else:
                            print("Switch to manual mode (press M) to use SPACE")
                    elif event.key == pygame.K_m:
                        # Toggle between manual and automatic evolution
                        self.manual_evolution = not self.manual_evolution
                        mode = "MANUAL" if self.manual_evolution else "AUTOMATIC"
                        print(f"️  Evolution mode: {mode}")
                    elif event.key == pygame.K_p:
                        # Plot statistics
                        self.plot_evolution_stats()
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        # Manually add more bees
                        added = self.add_new_bees(50)
                        print(f" Manually added {added:,} bees. Total: {len(self.bees):,}")
                    elif event.key == pygame.K_g:
                        # Adjust growth rate
                        growth_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
                        try:
                            current_idx = growth_rates.index(self.population_growth_rate)
                        except ValueError:
                            current_idx = 2  # Default to 0.3
                        self.population_growth_rate = growth_rates[(current_idx + 1) % len(growth_rates)]
                        print(f" Growth rate set to: {self.population_growth_rate*100:.0f}%")
                    elif event.key == pygame.K_c:
                        print(f"Maximum population is LOCKED at 5,000 bees")
            
            # Clear screen
            self.screen.fill(WHITE)
            
            # Update all bees
            for bee in self.bees:
                bee.update(self.pollen_sources, self.frame_count)
            
            # Draw pollen sources
            for source in self.pollen_sources:
                pygame.draw.circle(self.screen, source.color, 
                                 (int(source.x), int(source.y)), source.radius)
                pygame.draw.circle(self.screen, BLACK, 
                                 (int(source.x), int(source.y)), source.radius, 2)
            
            # Draw bees (optimize for large populations)
            if len(self.bees) <= 1000:
                # Draw all bees if population is manageable
                for bee in self.bees:
                    bee.draw(self.screen)
            else:
                # Draw only a sample for performance with large populations
                sample_size = min(500, len(self.bees))
                sample_bees = random.sample(self.bees, sample_size)
                for bee in sample_bees:
                    bee.draw(self.screen)
            
            # Draw UI
            self.draw_ui()
            
            # Evolution logic - only run automatically if not in manual mode
            if not self.manual_evolution and self.frame_count % self.generation_length == 0 and self.frame_count > 0:
                print(" Auto-advancing generation...")
                self.advance_generation_manually()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)
            self.frame_count += 1
        
        pygame.quit()
    
    def plot_evolution_stats(self):
        """Plot evolution statistics"""
        if not self.population_preferences_history:
            print(" No evolution data to plot yet. Advance some generations first!")
            return
        
        generations = list(range(len(self.population_preferences_history)))
        high_prefs = [prefs[0] for prefs in self.population_preferences_history]
        med_prefs = [prefs[1] for prefs in self.population_preferences_history]
        low_prefs = [prefs[2] for prefs in self.population_preferences_history]
        
        plt.figure(figsize=(15, 10))
        
        # Preferences evolution
        plt.subplot(2, 2, 1)
        plt.plot(generations, high_prefs, 'g-', label='High Quality', linewidth=3)
        plt.plot(generations, med_prefs, 'orange', label='Medium Quality', linewidth=3)
        plt.plot(generations, low_prefs, 'r-', label='Low Quality', linewidth=3)
        plt.xlabel('Generation')
        plt.ylabel('Average Preference')
        plt.title('Evolution of Pollen Preferences')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if self.generation_stats:
            # Average rewards
            plt.subplot(2, 2, 2)
            rewards = [stat['avg_reward'] for stat in self.generation_stats]
            gen_nums = [stat['generation'] for stat in self.generation_stats]
            plt.plot(gen_nums, rewards, 'b-', linewidth=3)
            plt.xlabel('Generation')
            plt.ylabel('Average Reward per Bee')
            plt.title('Average Reward Over Generations')
            plt.grid(True, alpha=0.3)
            
            # Population growth - MAX 5000
            plt.subplot(2, 2, 3)
            pop_sizes = [stat['population_size'] for stat in self.generation_stats]
            plt.plot(gen_nums, pop_sizes, 'purple', linewidth=3)
            plt.axhline(y=5000, color='red', linestyle='--', linewidth=2, label='Max: 5,000')
            plt.xlabel('Generation')
            plt.ylabel('Population Size')
            plt.title('Population Growth (Maximum: 5,000)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Current fitness distribution
            plt.subplot(2, 2, 4)
            if self.bees:
                fitness_values = [bee.lifetime_reward for bee in self.bees]
                plt.hist(fitness_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                plt.xlabel('Lifetime Reward')
                plt.ylabel('Number of Bees')
                plt.title(f'Fitness Distribution (Pop: {len(self.bees):,})')
                plt.grid(True, alpha=0.3)
        
        plt.suptitle(' BEE EVOLUTION ANALYTICS (Max Pop: 5,000) ', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
   
    print(" BEE SWARM EVOLUTION SIMULATION ")
   
    print()
    print(" MAXIMUM POPULATION: 5,000 BEES")
    print()
    print(" CONTROLS:")
    print("   SPACE → Advance Generation (Manual Mode)")
    print("   M → Toggle Manual/Auto Evolution") 
    print("   R → Reset Simulation")
    print("   + → Add 50 Bees Manually")
    print("   G → Cycle Growth Rate (10%-50%)")
    print("   C → Max pop locked at 5,000")
    print("   P → Plot Statistics")
    print("   ESC → Quit")
    print()
    print(" GROWTH PROJECTION (30% rate):")
    print("   Gen 0: 50 bees")
    print("   Gen 5: 186 bees") 
    print("   Gen 10: 693 bees")
    print("   Gen 15: 2,581 bees")
    print("   Gen 18: 5,000 bees (MAX)")
    print()
    print(" Population is HARD CODED to maximum 5,000")
    print("=" * 60)
    
    # Create and run simulation with 5000 max
    sim = BeeSwarmSimulation(num_bees=50)
    sim.run()
