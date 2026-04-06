# N.B. font is ANSI compact

MOTD_APP = r"""                                                                                                                 
█████▄  ▄▄▄  ▄▄    ▄▄      ▄████▄ ▄▄▄▄   ▄▄▄   ▄▄▄▄ ▄▄    ▄▄▄▄▄ 
██▄▄██ ██▀██ ██    ██      ██  ██ ██▄█▄ ██▀██ ██▀▀▀ ██    ██▄▄  
██▄▄█▀ ██▀██ ██▄▄▄ ██▄▄▄   ▀████▀ ██ ██ ██▀██ ▀████ ██▄▄▄ ██▄▄▄ 

v0.2.0 by wokaine                                                                                                                    
"""

MOTD_HGF = r"""                                                                                     
██  ██ ▄▄ ▄▄▄▄  ▄▄▄▄  ▄▄▄▄▄ ▄▄  ▄▄    ▄████  ▄▄▄▄▄ ▄▄   ▄▄   ██████ ▄▄ ▄▄  ▄▄ ▄▄▄▄  ▄▄▄▄▄ ▄▄▄▄  
██████ ██ ██▀██ ██▀██ ██▄▄  ███▄██   ██  ▄▄▄ ██▄▄  ██▀▄▀██   ██▄▄   ██ ███▄██ ██▀██ ██▄▄  ██▄█▄ 
██  ██ ██ ████▀ ████▀ ██▄▄▄ ██ ▀██    ▀███▀  ██▄▄▄ ██   ██   ██     ██ ██ ▀██ ████▀ ██▄▄▄ ██ ██                                                                                                                                                           
"""

MOTD_CC = r"""                                                                                                                                                                                                             
▄█████  ▄▄▄  ▄▄   ▄▄ ▄▄▄▄▄ ▄▄▄▄   ▄▄▄   ▄▄▄▄ ▄▄ ▄▄          
██     ██▀██ ██▀▄▀██ ██▄▄  ██▄██ ██▀██ ██▀▀▀ ██▄█▀          
▀█████ ▀███▀ ██   ██ ██▄▄▄ ██▄█▀ ██▀██ ▀████ ██ ██          
                                                       
▄█████  ▄▄▄  ▄▄     ▄▄▄▄ ▄▄ ▄▄ ▄▄     ▄▄▄ ▄▄▄▄▄▄ ▄▄▄  ▄▄▄▄  
██     ██▀██ ██    ██▀▀▀ ██ ██ ██    ██▀██  ██  ██▀██ ██▄█▄ 
▀█████ ██▀██ ██▄▄▄ ▀████ ▀███▀ ██▄▄▄ ██▀██  ██  ▀███▀ ██ ██                                                                                                                                        
"""

QUOTES = [("\"You have to remember that it's been raining.\"", "Taiwo Ogunlabi"), 
          ("\"Leave the football before the football leaves you.\"", "Jamie Carragher"), 
          ("\"This was the no-brainer, this was the banker, this was the one that couldn't fail, this was the one that's never failed.\"", "Gary Neville"), 
          ("\"He may have actually picked the wrong club.\"", "Jamie Carragher"),
          ("\"Yeah you've gone over there and you won a few trophies, but what your legacy is gonna be is that you are a Judas!\"", "Robbie Lyle"),
          ("\"It can.\"", "Mick McCarthy"),
          ("\"You've got to die to get three points!\"", "Neil Warnock"),
          ("\"One word: pure class.\"", "Adebayo Akinfenwa"),
          ("\"If I speak I am in big trouble.\"", "Jose Mourinho"),
          ("\"I will be there no matter what.\"", "Kylian Mbappe)")
]

ABOUT_MAIN = """
Welcome to the [b]Ball Oracle project[/b], a collection of football-themed machine learning projects.

Data used for the Hidden Gem Finder comes from [link="https://understat.com/"]Understat[/link] using the [link="https://github.com/collinb9/understatAPI"]understat API[/link] project v0.7.0 by collinb9.

Data used for the Comeback Calculator comes from [link="https://football-data.co.uk/"]Football Data[/link].

[i]Use the menu below to launch the available programs[/i].
"""

ABOUT_HGF = """
Welcome to the [b]Hidden Gem Finder[/b], a program that groups players by similar playstyles to allow you to find underrated players!

This project in particular uses [link="https://en.wikipedia.org/wiki/K-means_clustering"]K-means Clustering[/link], a machine learning technique that groups datapoints based on their location in the feature space.

Feel free to have a play around with the options!
"""

ABOUT_CC = """
Welcome to the [b]Comeback Calculator[/b], a ML model that predicts whether or not a team will come back to win after trailing at half time!

This project uses the [link="https://en.wikipedia.org/wiki/XGBoost"]XGBoost[/link] library to classify whether or not a comeback will happen.

The model still needs a bit of tweaking, but here's the results from the games so far this season!
"""

