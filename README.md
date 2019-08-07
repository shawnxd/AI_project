# AI_project
Term project for CIS 521 Artificial Intelligence
Videos for the project here: https://youtu.be/NsPaquG9qA8

Download the sphero-project normally like how it is outlined in the course instructions. In sphero-project —> src, we are going to add another file. We are going to add the r2d2_final.py and the glove.6B.300d.magnitude files. Make sure to download the .magnitude file from the pymagnitude website (https://github.com/plasticityai/magnitude)

Then, like normal, do the terminal commands that will allow you to use the python interpreter to control the robot. I used python3 interpreter and had to install some modules like the following before I could do this:

pip install nltk

pip install sklearn

pip install pymagnitude 

However, if this is not a problem and you already have these modules (and any other necessary modules) installed, then we can proceed with setting up the r2d2 Command Center! 

In the python environment, type in the following commands: 

from client import DroidClient

droid = DroidClient() 

droid.scan() # Scan for droids.

droid.connect_to_droid('D2-33C3') # droid ID here

Now we have connected to the robot… this is where our r2d2_final.py script comes into play. Type in the following commands: 

from r2d2_final import CommandCenter

command = CommandCenter(droid)

command.run_commands()

Feel free to play with the robot now!
