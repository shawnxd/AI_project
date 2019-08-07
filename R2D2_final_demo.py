############################################################
# CIS 521:  R2D2 Group Project — Give R2D2 Commands using NLP!
############################################################
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pymagnitude import *
from client import DroidClient
import random

#this class acts as a command center for the droids!
class CommandCenter:
    def __init__(self, droid):
        # using the code from Demo of Word Vectors for MindCore 2019, we can use pymagnitude to load the glove file quickly

        #when running in the command line, we will pass in a DroidClient object
        self.droid = droid

        self.vectors = Magnitude("glove.6B.300d.magnitude")

        # use vectors.query('string here') to get the associated word vector
        # use the vectors.similarity function to get the similarity between two vectors

        # build the lists for the words we will use
        sleep = ['sleep', 'good night', 'have some rest', 'stop working', 'exhausted', 'tired', 'lie down',
                 'please just rest',
                 'have a good dream', 'you may cease working', 'leave the work alone', 'close your eyes', 'nice dream',
                 'take a nap', 'you are done', 'you have finished all your work', 'go to sleep', 'take some rest',
                 'relax now', 'relax for a moment', 'loosen up', 'take a break', 'time for a break',
                 'time for relaxation',
                 'go to sleep']

        wake = ['wake up', 'get up', 'start working', 'new work has been assigned to you', 'your rest time is now over',
                'start your business', 'begin working', 'start doing something', 'you are on duty', 'good morning',
                'wake up now', 'open your eyes', 'time to start']

        turn_left = ['turn left', 'when facing north turn to the west', 'when facing south turn to the east',
                     'when facing east, turn to north', 'when facing west, turn to south',
                     'counter clockwise ninety degrees',
                     'face your left side, please']

        turn_right = ['turn right', 'when facing north turn to the east', 'when facing south turn to the west',
                      'when facing east turn to the south', 'when facing west turn to the north',
                      'clockwise ninety degrees',
                      'face your right side please']

        waddle = ['waddle', 'walk', 'roam', 'stride', 'take a step', 'walk slowly']

        drive = ['enter drive mode', 'start driving', 'start rolling', 'begin to roll', 'begin to drive', 'run fast',
                 'start moving fast', 'move as fast as you can', 'run with your wheels']

        sound = ['play sound', 'make some noises', 'sing a song', 'talk to me', 'have a conversation', 'give a speech',
                 'say something', 'start talking', 'speak loudly', 'sing for me', 'speak to me']

        light_blue = ["change the light color to blue"]
        light_red = ["change the light color to red"]
        light_green = ["change the light color to green"]

        no_action = ['no action', 'class']

        # indices
        # sleep = 0
        # wake = 1
        # turn_left = 2
        # turn_right = 3
        # waddle = 4
        # drive = 5
        # sound = 6
        # light_blue = 7
        # light_red = 8
        # light_green = 9
        # no_action = 10

        self.sw_commands = [sleep, wake, turn_left, turn_right, waddle, drive, sound, light_blue, light_red, light_green,
                       no_action]
        self.stop_words = stopwords.words('english')


    #each sentence should be given as strings
    #helper function 1 that compares the similarity of strings
    def get_similarity(self, sentence1, sentence2):
        s1 = sentence1.lower()
        s2 = sentence2.lower()
        sw_token_s1 = word_tokenize(s1)
        sw_token_s2 = word_tokenize(s2)

        #remove the stop words
        token_s1 = [word for word in sw_token_s1 if word not in self.stop_words]
        token_s2 = [word for word in sw_token_s2 if word not in self.stop_words]

        #get the average of both sentences vectors

        length = len(self.vectors.query('you'))
        sum1_vector = np.zeros(length)
        for word1 in token_s1:
            vector1 = self.vectors.query(word1)
            sum1_vector = sum1_vector + vector1
        sum1_vector = sum1_vector / len(token_s1)

        sum2_vector = np.zeros(length)
        for word2 in token_s2:
            vector2 = self.vectors.query(word2)
            sum2_vector = sum2_vector + vector2
        sum2_vector = sum2_vector / len(token_s2)

        #return the cosine similarity between the averaged vectors
        similarity = self.vectors.similarity(sum1_vector, sum2_vector)
        return similarity


    #helper function 2 that takes in a list of command sentences and compares it to target sentence
    #and returns a list of the cosine similarities
    def compare_command(self, command_list, input_sentence):
        similarities = []
        for sentence in command_list:
            similarity = self.get_similarity(sentence, input_sentence)
            similarities.append(similarity)
        return similarities

    #this function takes in an input command sentence and uses compare_command to find the command with the biggest similarity
    def pick_command(self, input):

        index = 0
        max_index = 0

        #keeps track of maximum similarity
        maximum = float('-inf')

        for command_list in self.sw_commands:
            similarity_list = self.compare_command(command_list, input)
            list_max = max(similarity_list)

            if list_max > maximum:
                maximum = list_max
                max_index = index
            index = index + 1

        #now, we have a track of the max index (the index with max similarity)...
        #the indices are ordered like so
        # indices
        # sleep = 0
        # wake = 1
        # turn_left = 2
        # turn_right = 3
        # waddle = 4
        # drive = 5
        # sound = 6
        # light_blue = 7
        # light_red = 8
        # light_green = 9
        # no_action = 10

        #therefore, we will pick a command based off of these

        index = max_index

        if index == 0:
            self.droid.sleep()
            return ('self.droid.sleep()', 'sleep')
        elif index == 1:
            self.droid.wake()
            return ('self.droid.wake()', 'wake')
        elif index == 2:
            self.droid.turn(270)
            return ('self.droid.turn(270)', 'turn_left')
        elif index == 3:
            self.droid.turn(270)
            return ('self.droid.turn(270)', 'turn_right')
        elif index == 4:
            self.droid.set_waddle(True)
            return ('self.droid.set_waddle(True)', 'waddle')
        elif index == 5:
            self.droid.enter_drive_mode()
            return ('self.droid.enter_drive_mode()', 'drive')
        elif index == 6:
            #generate a random integer between 1 and 10
            soundID = random.randint(1, 11)
            self.droid.play_sound(soundID)
            return ('self.droid.play_sound(soundID)', 'sound')
        elif index == 7:
            self.droid.set_front_LED_color(0, 0, 255)
            self.droid.set_back_LED_color(0, 0, 255)
            return ('self.droid.set_front_LED_color(0, 0, 255), self.droid.set_back_LED_color(0, 0, 255)', 'light_blue')
        elif index == 8:
            self.droid.set_front_LED_color(255, 0, 0)
            self.droid.set_back_LED_color(255, 0, 0)
            return ('self.droid.set_front_LED_color(255, 0, 0), self.droid.set_back_LED_color(255, 0, 0)', 'light_red')
        elif index == 9:
            self.droid.set_front_LED_color(0, 255, 0)
            self.droid.set_back_LED_color(0, 255, 0)
            return (' self.droid.set_front_LED_color(0, 255, 0), self.droid.set_back_LED_color(0, 255, 0)', 'light_green')
        elif index == 10:
            return ('no action', 'no_action')


    def run_commands(self):
        #run the actual program
        running = True
        while running:
            print('Hello, welcome to the R2D2 command center! Give your droid a command today. If you do not want to give it '
                  'a command, then enter "Exit" ')
            command = input('Your command: ')

            if command.lower() == 'exit':
                print('R2D2 is done now!')
                running = False
                break;

            command_to_give = self.pick_command(command)
            print('You have told R2D2 to perform the following action: ', command_to_give[1])
            print(command_to_give[0])





