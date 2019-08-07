# AI_project
Term project for CIS 521 Artificial Intelligence.
Videos for the project here: https://youtu.be/NsPaquG9qA8
{\rtf1\ansi\ansicpg1252\cocoartf1671
{\fonttbl\f0\fnil\fcharset0 HelveticaNeue;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab560
\pard\pardeftab560\slleading20\partightenfactor0

\f0\fs24 \cf0 Instructions for the r2d2 project: \
\
Download the sphero-project normally like how it is outlined in the course instructions. In sphero-project \'97> src, we are going to add another file. We are going to add the r2d2_final.py and the glove.6B.300d.magnitude files. \
\pard\pardeftab560\slleading20\pardirnatural\partightenfactor0
\cf0 \
\pard\pardeftab560\slleading20\partightenfactor0
\cf0 Then, like normal, do the terminal commands that will allow you to use the python interpreter to control the robot. I used python3 interpreter and had to install some modules like the following before I could do this: \
pip install nltk\
pip install sklearn\
pip install pymagnitude \
\
However, if this is not a problem and you already have these modules (and any other necessary modules) installed, then we can proceed with setting up the r2d2 Command Center! \
\pard\pardeftab560\slleading20\pardirnatural\partightenfactor0
\cf0 \
\pard\pardeftab560\slleading20\partightenfactor0
\cf0 In the python environment, type in the following commands: \
from client import DroidClient\
droid = DroidClient() \
droid.scan() # Scan for droids.\
droid.connect_to_droid('D2-33C3') # droid ID here\
\
Now we have connected to the robot\'85 this is where our r2d2_final.py script comes into play. Type in the following commands: \
\
from r2d2_final import CommandCenter\
command = CommandCenter(droid)\
command.run_commands()\
\
Now, the terminal should be prompting for you to type in your commands! This is great! You now have command of r2d2\'92s CommandCenter. You can type in whatever you want and see how r2d2 reacts to it. Remember to type \'93exit\'94 to leave the program! }
