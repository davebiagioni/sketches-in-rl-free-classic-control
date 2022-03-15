import sys

import gym
import pygame


# Give the user instructions before playing
print(
    """
    Try to get the pendulum upright and keep it there as long as possible!
    The control action is the torque on the fulcrum point, with values in 
    [-2.0, 2.0].  Use the LEFT, RIGHT, and DOWN arrow keys to adjust the
    torque.  These have the effect:
        LEFT:  decrease by 0.5 (clockwise)
        RIGHT: increase by 0.5 (counter-clockwise)
        DOWN:  set to 0
    So, to get the max negative torque possible, you'll tap the left key 4
    times in a row.  Your final "reward" will print out at the end of the 
    episode.  
    
    Happy torquing!
    """)

input("Press ENTER when ready")

# Initialize pygame and display
pygame.init()
display = pygame.display.set_mode((300, 300))

# Initialize the gym environment
env = gym.make("Pendulum-v1")
env.reset()
reward = 0.    # total reward
action = 0.    # initial action
done = False   # done flag to know when episode is over
# Main event loop
while not done:
       
    # creating a loop to check events that
    # are occuring
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
         
    # checking if keydown event happened or not
    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        action -= 0.5
        action = max(action, -2.0)
    
    if keys[pygame.K_RIGHT]:
        action += 0.5
        action = min(action, 2.0)
    
    if keys[pygame.K_DOWN]:
        action = 0
                
    _, r, done, _ = env.step([action])
    reward += r
    env.render()

env.close()

print("reward", reward)
