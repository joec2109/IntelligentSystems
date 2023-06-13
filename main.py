import pygame
import random
import numpy as np

mazeArray = np.array([[1.0] * 33 for i in range(33)])

possibleStartPosList = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29]

startingPos = [random.choice(possibleStartPosList),0]

mazeArray[startingPos[0]][startingPos[1]] = 2

mazeArray[startingPos[0]][startingPos[1] + 1] = 0

currentStartingPos = [startingPos[0], startingPos[1]+1]

possibleCoords = []

randomFinish = ()

# Draw the elements in the maze
def drawPath():
    # Loop through each element in mazeArray to determine whether they are a block or a space
    for i in range(0,len(mazeArray)):
        for j in range(0,len(mazeArray[i])):
            if mazeArray[i][j] == 1:
                newRect = pygame.draw.rect(screen, (0,0,0), pygame.Rect(i * 16, j * 16, 16, 16))
            elif mazeArray[i][j] == 2:
                newRect = pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(i * 16, j * 16, 16, 16))
                newRect = pygame.draw.circle(screen, (255, 0, 0), ((i * 16)+8, (j * 16)+8), 4)
            elif mazeArray[i][j] == 3:
                newRect = pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(i * 16, j * 16, 16, 16))
            else:
                newRect = pygame.draw.rect(screen, (255,255,255), pygame.Rect(i * 16, j * 16, 16, 16))

    screen.blit(pygame.transform.rotate(screen, 90), (0, 0))
    pygame.display.update()

# Generate the maze randomly by creating random paths
def genMaze():
    # If the tiles 2 spaces to the right/left/up/down of the starting position are a wall, a path can be made to it
    if currentStartingPos[1] + 2 < 32: # '< 32' and '> 0' ensures the path cannot go outside the maze outer walls
        if mazeArray[currentStartingPos[0]][currentStartingPos[1] + 2] == 1:
            possibleCoords.append([currentStartingPos[0], currentStartingPos[1] + 2, "Right"])
    if currentStartingPos[1] - 2 > 0:
        if mazeArray[currentStartingPos[0]][currentStartingPos[1] - 2] == 1:
            possibleCoords.append([currentStartingPos[0], currentStartingPos[1] - 2, "Left"])
    if currentStartingPos[0] + 2 < 32:
        if mazeArray[currentStartingPos[0] + 2][currentStartingPos[1]] == 1:
            possibleCoords.append([currentStartingPos[0] + 2, currentStartingPos[1], "Up"])
    if currentStartingPos[0] - 2 > 0:
        if mazeArray[currentStartingPos[0] - 2][currentStartingPos[1]] == 1:
            possibleCoords.append([currentStartingPos[0] - 2, currentStartingPos[1], "Down"])

    # Whilst there are options for the maze path to be made in, continue with them
    while len(possibleCoords) > 0:
        # Choose a random possible path
        chosenCoord = random.randint(0, len(possibleCoords) - 1)

        # If the random possible path tile is a wall, then it can be set to an open space
        if mazeArray[possibleCoords[chosenCoord][0]][possibleCoords[chosenCoord][1]] == 1:
            mazeArray[possibleCoords[chosenCoord][0]][possibleCoords[chosenCoord][1]] = 0

            # Once a move is made, make sure to set the tile before it to an open space too.
            if (possibleCoords[chosenCoord][2] == "Right") and mazeArray[possibleCoords[chosenCoord][0]][possibleCoords[chosenCoord][1] - 1] == 1:
                mazeArray[possibleCoords[chosenCoord][0]][possibleCoords[chosenCoord][1] - 1] = 0
            elif (possibleCoords[chosenCoord][2] == "Left") and mazeArray[possibleCoords[chosenCoord][0]][possibleCoords[chosenCoord][1] + 1] == 1:
                mazeArray[possibleCoords[chosenCoord][0]][possibleCoords[chosenCoord][1] + 1] = 0
            elif (possibleCoords[chosenCoord][2] == "Up") and mazeArray[possibleCoords[chosenCoord][0] - 1][possibleCoords[chosenCoord][1]] == 1:
                mazeArray[possibleCoords[chosenCoord][0] - 1][possibleCoords[chosenCoord][1]] = 0
            elif (possibleCoords[chosenCoord][2] == "Down") and mazeArray[possibleCoords[chosenCoord][0] + 1][possibleCoords[chosenCoord][1]] == 1:
                mazeArray[possibleCoords[chosenCoord][0] + 1][possibleCoords[chosenCoord][1]] = 0
            pygame.display.flip()
            pygame.event.pump()
            pygame.time.delay(1)
            drawPath()

        currentPos = [possibleCoords[chosenCoord][0], possibleCoords[chosenCoord][1]]   # Update current position

        possibleCoords.pop(chosenCoord) # Make sure to remove the move made

        # Check for more possible moves
        if currentPos[1] + 2 < 33:
            if mazeArray[currentPos[0]][currentPos[1] + 2] == 1 and mazeArray[currentPos[0]][currentPos[1] + 1] == 1:
                possibleCoords.append([currentPos[0], currentPos[1] + 2, "Right"])
        if currentPos[1] - 2 > 0:
            if mazeArray[currentPos[0]][currentPos[1] - 2] == 1 and mazeArray[currentPos[0]][currentPos[1] - 1] == 1:
                possibleCoords.append([currentPos[0], currentPos[1] - 2, "Left"])
        if currentPos[0] + 2 < 32:
            if mazeArray[currentPos[0] + 2][currentPos[1]] == 1 and mazeArray[currentPos[0] + 1][currentPos[1]] == 1:
                possibleCoords.append([currentPos[0] + 2, currentPos[1], "Up"])
        if currentPos[0] - 2 > 0:
            if mazeArray[currentPos[0] - 2][currentPos[1]] == 1 and mazeArray[currentPos[0] - 1][currentPos[1]] == 1:
                possibleCoords.append([currentPos[0] - 2, currentPos[1], "Down"])

pygame.init()

# Set up the drawing window
screen = pygame.display.set_mode([33*16, 33*16])

if __name__ == '__main__':
    # Call the generate maze function to create the maze.
    genMaze()

    # Find a position to place the finish block to complete the maze
    possibleFinishes = []
    ''' Loop through each last element in each list within the multidimensional 'mazeArray' to detect if the finish 
    block can be accessed. '''
    for i in range(len(mazeArray)-1):
        if mazeArray[i][31] == 0:
            possibleFinishes.append([i, 32])

    # Select a random finish block
    randomFinish = random.choice(possibleFinishes)

    # Set the finish block to 3
    mazeArray[randomFinish[0]][randomFinish[1]] = 3

    drawPath()

    # Initialise the q table for measuring how good each move in the maze is
    q_table_shape = tuple(list(mazeArray.shape) + [4])
    q_table = np.random.uniform(low=0, high=0.1, size=q_table_shape)

    # Define the rewards for each state
    rewards = np.zeros_like(mazeArray)
    rewards[mazeArray == 1] = -1  # Assign a negative reward to wall states
    rewards[mazeArray == 2] = 0 # Assign a zero reward to the start state
    rewards[mazeArray == 3] = 1  # Assign a positive reward to the goal state
    rewards[mazeArray == 0] = 0

    # Define the learning rate, discount factor, and exploration rate
    learning_rate = .8
    discount_factor = 0.99
    exploration_rate = .4

    # Define the maximum number of episodes and steps per episode
    max_episodes = 10000
    max_steps_per_episode = 200

    # optimal_path contains the coordinates of the best path to the goal state
    optimal_path = []

    # Counter used to detect if the AI has found a path to the finish tile multiple times.
    path_counter = 0

    for episode in range(max_episodes):
        # Reset the environment to the start state
        state = startingPos
        episode_reward = 0
        optimal_path = []
        # Loop over steps
        for step in range(max_steps_per_episode):
            if episode != max_episodes-1:
                for i in range(len(mazeArray)):
                    for j in range(len(mazeArray[i])):
                        if mazeArray[i][j] == 2:
                            mazeArray[i][j] = 0

            # Choose an action based on the current state using an epsilon-greedy strategy
            ru = random.uniform(0,1)
            if ru < exploration_rate:
                action = random.randint(0, 3)
            else:
                action = np.argmax(q_table[tuple(state)])
            # Take the chosen action and observe the next state and reward
            next_state = tuple(
                (np.array(state) + np.array([-1, 0, 1, 0, 0, -1, 0, 1]).reshape((4, 2))[action]).astype(int))

            if mazeArray[next_state[0]][next_state[1]] == 1 or rewards[next_state] == 1:
                done = True
            else:
                done = False
            reward = rewards[next_state]

            # Update the Q-value of the current state-action pair using the Q-learning formula (Bellman equation)
            current_q = q_table[tuple(state)][action]
            next_q = np.max(q_table[next_state])
            new_q = current_q + learning_rate * (reward + discount_factor * next_q - current_q)
            q_table[tuple(state)][action] = new_q

            # Update the episode reward and current state
            episode_reward += reward
            state = next_state

            if episode == max_episodes-1:
                mazeArray[state[0]][state[1]] = 2

            optimal_path.append([(state[0]), (state[1])])

            if rewards[next_state] == 1:
                path_counter += 1

            print(f"Episode: {episode} Taking action {action} at state {state} step: {step} score: {episode_reward}")

            # If the agent reaches the goal state or a wall, break out of the loop
            if done:
                break

            mazeArray[state[0]][state[1]] = 2

            # Uncomment the line below to make the program display the positions being tested (works faster without)
            drawPath()

        # Print the episode number and episode reward
        print(f"Episode {episode + 1} completed with reward {episode_reward}")

        ''' If the agent has found its way to the finish tile 50 times, set the tiles in the path it has found to it to
        2 so that the path is displayed. Also break the 'episode' loop to stop the Q learning '''
        if path_counter == 50:
            for i in optimal_path:
                mazeArray[i[0]][i[1]] = 2
            break

        # Decay the exploration rate
        exploration_rate *= 0.99

    # Ensure the starting position is set to 2
    mazeArray[startingPos[0]][startingPos[1]] = 2
    drawPath()


# Run until the user asks to quit
running = True
while running:

    # If window close button clicked, end the simulation
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background with white
    screen.fill((255, 255, 255))
    for i in range(0,len(mazeArray)):
        for j in range(0,len(mazeArray[i])):
            if mazeArray[i][j] == 1:
                newRect = pygame.draw.rect(screen, (0,0,0), pygame.Rect(i * 16, j * 16, 16, 16))
            elif mazeArray[i][j] == 2:
                newRect = pygame.draw.circle(screen, (255, 0, 0), ((i * 16) + 8, (j * 16) + 8), 4)
            elif mazeArray[i][j] == 3:
                newRect = pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(i * 16, j * 16, 16, 16))
            else:
                newRect = pygame.draw.rect(screen, (255,255,255), pygame.Rect(i * 16, j * 16, 16, 16))


    screen.blit(pygame.transform.rotate(screen, 90), (0, 0))
    pygame.display.update()

pygame.quit()
