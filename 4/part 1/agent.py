import numpy as np
import utils
import random


class Agent:

    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

        self.reset()

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''
        #get corresponding current state number
        snake_head_x = state[0]
        snake_head_y = state[1]
        snake_body = state[2]
        food_x = state[3]
        food_y = state[4]

        num0, num1, num2, num3, num4, num5, num6, num7 = 0, 0, 0, 0, 0, 0, 0, 0
        if snake_head_x == 1*utils.GRID_SIZE:
            num0 = 1
        elif snake_head_x == 560-2*utils.GRID_SIZE:
            num0 = 2

        if snake_head_y == 1*utils.GRID_SIZE:
            num1 == 1
        elif snake_head_y == 560-2*utils.GRID_SIZE:  #??? 11 or 480 or 520 or 560???
            num1 = 2

        if snake_head_x == food_x:
            num2 = 0
        elif snake_head_x > food_x:
            num2 = 1
        else:
            num2 = 2

        if snake_head_y == food_y:
            num3 = 0
        elif snake_head_y > food_y:
            num3 = 1
        else:
            num3 = 2

        for seg in snake_body:
            if seg[0] == snake_head_x and seg[1] + 1*utils.GRID_SIZE == snake_head_y:
                num4 = 1
            if seg[0] == snake_head_x and seg[1] - 1*utils.GRID_SIZE == snake_head_y:
                num5 = 1
            if seg[0] + 1*utils.GRID_SIZE == snake_head_x and seg[1] == snake_head_y:
                num6 = 1
            if seg[0] - 1*utils.GRID_SIZE == snake_head_x and seg[1] == snake_head_y:
                num7 = 1

        #update Q-Table only if previous action is not None
        if self.a != None and self.s != None:
            prev_snake_head_x = self.s[0]
            prev_snake_head_y = self.s[1]
            prev_snake_body = self.s[2]
            prev_food_x = self.s[3]
            prev_food_y = self.s[4]

            prev_action = self.a
            if points > self.points:
                reward = 1
            elif dead:
                reward = -1
            else:
                reward = -0.1

            #find the previous state numbers
            prevNum0, prevNum1, prevNum2, prevNum3, prevNum4, prevNum5, prevNum6, prevNum7 = 0, 0, 0, 0, 0, 0, 0, 0
            if prev_snake_head_x == 1*utils.GRID_SIZE:
                prevNum0 = 1
            elif prev_snake_head_x == 560-2*utils.GRID_SIZE:
                prevNum0 = 2

            if prev_snake_head_y == 1*utils.GRID_SIZE:
                prevNum1 == 1
            elif prev_snake_head_y == 560-2*utils.GRID_SIZE:  #??? 11 or 480 or 520 or 560???
                prevNum1 = 2

            if prev_snake_head_x == prev_food_x:
                prevNum2 = 0
            elif prev_snake_head_x > prev_food_x:
                prevNum2 = 1
            else:
                prevNum2 = 2

            if prev_snake_head_y == prev_food_y:
                newNum3 = 0
            elif prev_snake_head_y > prev_food_y:
                prevNum3 = 1
            else:
                prevNum3 = 2

            for seg in prev_snake_body:
                if seg[0] == prev_snake_head_x and seg[1] + 1*utils.GRID_SIZE == prev_snake_head_y:
                    prevNum4 = 1
                if seg[0] == prev_snake_head_x and seg[1] - 1*utils.GRID_SIZE == prev_snake_head_y:
                    prevNum5 = 1
                if seg[0] + 1*utils.GRID_SIZE == prev_snake_head_x and seg[1] == prev_snake_head_y:
                    prevNum6 = 1
                if seg[0] - 1*utils.GRID_SIZE == prev_snake_head_x and seg[1] == prev_snake_head_y:
                    prevNum7 = 1

            #calculate alpha
            alpha = self.C/(self.C+self.N[prevNum0][prevNum1][prevNum2][prevNum3][prevNum4][prevNum5][prevNum6][prevNum7][prev_action])
            maxQValue = max(self.Q[num0][num1][num2][num3][num4][num5][num6][num7])
            prevQValue = self.Q[prevNum0][prevNum1][prevNum2][prevNum3][prevNum4][prevNum5][prevNum6][prevNum7][prev_action]
            self.Q[prevNum0][prevNum1][prevNum2][prevNum3][prevNum4][prevNum5][prevNum6][prevNum7][prev_action] += alpha*(reward + self.gamma*maxQValue - prevQValue)

        #only choose action if not dead
        if not dead:
            action = 0
            fValue = float("-inf")
            fValues = []
            for i in range(4):
                if self.N[num0][num1][num2][num3][num4][num5][num6][num7][i] < self.Ne:
                    fValues.append(1)
                else:
                    fValues.append(self.Q[num0][num1][num2][num3][num4][num5][num6][num7][i])

            for i in range(3,-1,-1):
                if fValues[i] > fValue:
                    action = i
                    fValue = fValues[i]

            #update the self.N
            self.N[num0][num1][num2][num3][num4][num5][num6][num7][action] += 1
            self.a = action
            self.s = state
            self.s[2] = list(state[2])
            self.points = points
            return action

        self.reset()
        return 0
