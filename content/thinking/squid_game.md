---
title: "Thoughts on - The Squid Game (spoilers for Ep.7)"
date: 2021-10-08T01:45:16-04:00
categories: ["Essay"]
tags: ["Movie", "Math"]
draft: false
---

> In this game, 16 players attempt cross a chasm by traversing a glass bridge with 18 steps. 
>
> Each step consists of a pair of glass panes. One is tempered glass that can support the weight of two players. The other is regular glass that will shatter when stepped on, causing that player to fall to their death. 
>
> At each step the panes are randomized, and the players have no way of knowing which is which until they jump on one. Each step is essentially a 50/50 guess, and all 18 steps must be crossed in order.

### Question 1 - Expected Number of Survivors

The first thing came to my mind after watching this episode was, 
> **how many pairs of glass panes do I need  if I want X number of players to make it?** >

This is important because there will be only one final game left to win the price, so one will need to make sure there will be enough player who makes it but not too many. The above question is equal to 

>**what is the expected number of survivors among X players crossing Y pair of glass?**

 Lets first make some assumption, 

- No one will suicide (suicide by jumping off the bridge will not reveal additional information for other players) 
- Everyone jumps randomly with 50/50 guess.
- No one will push other people down the bridge using violence.
- Following player can remember previous players' choices and make rational choices.

Under all these assumption, this game can be simplified as *tossing a fair coin `$Y$`  times and counting # of tails. The game will terminate earlier if  # of tails reaches `$X$`. Each tail event represents the death of one player and the next player will start from where the previous one died.*

Lets take an example, the sequence  "`$+ + + -  + - + +$`" describes the following scenario, 1st player moved 3 steps forward and died at the 4th pane, 2nd player continue from the 4th to the 5th and fell off at the 6th, the 3rd player continue from 6th pane and successfully pass the 7th and 8th. Since the 3rd player already passed, the rest of the players will just follow through.

If `$X \geq Y$`, we know the number of survivors will be `$ X - 0.5Y$`, where `$0.5Y$`is the mean of `$B(Y, 0.5)$` distribution. When `$ X < Y$`, Binomial distribution will be truncated by earlier termination. 

Roughly speaking, with 16 players and 18 pair of glass panes, the average number of death should be little bit less than 9 therefore the number of survivors will be a little more than 7.

### Question 2 - Survival Probability of `$i^{th}$` player

The next question came to my mind is **what is the survival probability of the  `$i^{th}$` player?** 

The survival probability of the `$i^{th}$` player is equal to `$P(t \leq i - 1)$`, where `$ t \sim B(Y, 0.5)$`.

```python
from scipy.stats import binom
for i in range(1, 17):
    prob = binom.cdf(i - 1, 18, 0.5) * 100
    print('%s-th player survival prob = %.2f %%' % (i, prob))

1-th player survival prob = 0.00 %
2-th player survival prob = 0.01 %
3-th player survival prob = 0.07 %
4-th player survival prob = 0.38 %
5-th player survival prob = 1.54 %
6-th player survival prob = 4.81 %
7-th player survival prob = 11.89 %
8-th player survival prob = 24.03 %
9-th player survival prob = 40.73 %
10-th player survival prob = 59.27 %
11-th player survival prob = 75.97 %
12-th player survival prob = 88.11 %
13-th player survival prob = 95.19 %
14-th player survival prob = 98.46 %
15-th player survival prob = 99.62 %
16-th player survival prob = 99.93 %
```

### Question 3 - Optimal Strategy

Knowing all these, is it possible to improve our chance of survival? For the first 4 players, it is almost impossible to survive and the only way out is for the half of the group to VETO the game. But we also observed that, the chance of the 9-th player (the vote we need for majority vote) to survive is significantly higher, jumping from 24% to 40%, and he/she highly likely will take the chance.

BUT, all our analysis are based on the assumption that the player will jump and as long as they jump, successful or not, will reveal useful information for other player behind them. What if they commit suicide? If the first 4 players jump off the bridge, the odds for the following players will change significantly. And for the first 4 players, the odds of survival are so low that if I were them I probably will be indifferent between the two options. And the only difference between my choices will be **if I want others to benefit from my death or not.**

So maybe,  the first four people can group together and try to negotiate a deal with the 5-9 players to VETO the game.



