## Contents
- [Contents](#contents)
- [BALL ORACLE v0.1.0](#ball-oracle-v010)
  - [About](#about)
  - [Hidden Gem Finder](#hidden-gem-finder)
  - [Executing](#executing)
  - [What's Next?](#whats-next)

## BALL ORACLE v0.1.0
A collection of football themed machine learning projects presented in a terminal UI.

Author: Freddie Butterfield (wokaine)

![Image of home screen](img/BallOracle_2026-02-04T22_02_03_873285.svg)

### About
For a while I was struggling to motivate myself to develop programs that were beyond what I was already doing at university. Then I had the realisation to combine my interests with computer science, and this project is the result of that.

Ball oracle is a collection of various football analysis programs powered by fundamental machine learning concepts. Currently only one has been implemented: Hidden Gem Finder. I have plans to implement many more (so long as university does not get in the way).

The aim of this is not to provide an amazing tool for football clubs to use, but to rather employ my knowledge of machine learning on to a subject I'm quite interested in. I am limited by the fact that my football analysis knowledge is next-to-none and that I am using data from free sources, both of which will affect the performance and predictions that the tools provide.

The data source I am using is [Understat](https://understat.com/) which supplies rudimentary data from 5 European leagues:
- Premier League (EPL)
- La Liga
- Bundesliga
- Serie A
- Ligue 1
- Russian Football Premier League

This is nicely wrapped up in a [library](https://github.com/collinb9/understatAPI) by user collinb9, so shout out to him!

[Textual](https://github.com/Textualize/textual) is used for the UI of the project.

### Hidden Gem Finder
![Image of Hidden Gem Finder](img/BallOracle_2026-02-04T22_02_55_829795.svg)

The Hidden Gem Finder aims to group players into clusters based on the features provided by Understat through [K-means Clustering](https://en.wikipedia.org/wiki/K-means_clustering). You can then search for a player and analyse the cluster to find players who have a similar play style but are maybe a bit more underrated or underutilised by their team, which could be potentially useful from a recruitment perspective.

### Executing
At this point in time I have not implemented any way for you to easily test this on your own. Though if you have Python installed and you are desperate you can always download the zip and run it through the terminal:

Install requirements
```
python -m pip install -r requirements.txt
```

Run
```
python run app.py
```
or
```
textual run app.py
```

### What's Next?
There's a lot of things to be done, especially very boring things.
- [ ] Docstrings
- [ ] Appropriate testing
- [ ] Comeback calculator
- [ ] Market value predictor

What you are seeing at the moment is a rough version that I'm happy to show to you, and I hope that I can continue to find the motivation to work on this project! 
