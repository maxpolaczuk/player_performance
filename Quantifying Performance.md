# On form: *estimation of a players upcoming performance.*

Estimation of the outcome of a game requires knowledge and insight into many different things (variables). Understand all variables that affect the final outcome of a game and you will know for certain what will happen.  If only such a task was so simple...\
The crux of trying to bet on the outcome of a game (or anything in life with uncertainty) is that the amount of variables that contribute to the final outcome of a series of actions is infinitely many and thus infinitely complex to measure. The term 'estimation' comes from approximating reality- we do not know all these variables and want to extract the most important ones in order to make an estimation of what will happen. We believe that one of these important variables to estiamte is 'player form'.
**What is 'player form'?** \
In short, form is a measure of how well a player is doing as compared to how he normally does. People sometimes describe a player being on form for long periods of time as a *hot-streak*, while conversly describing a player who is not on form as *in a slump*. \
More formally for a single game,  we measure form as the percentage deviation from their mean (average) performance.
We argue (and hope you can agree) that understanding when a player is (or isn't) on form should give us some more confidence as to who is going to win the game. 

**Estimation of player form** \
One thing you may be thinking to yourself is:

*'Why do this? I am immersed in the CS:GO communinity, I watch all the games, I'm on CS:GO Reddit every day-I know if a player is on form or not'*. 

This may but true, but can you give this a number? The problem we are solving is being able to **quantify** form- to put a number on it and to be able to code this as inputs to a Machine Learning algorithm. Putting a number on form, not only makes this tangible, but makes comparison of form between different players trivial.

**So without futher adieu, let us talk about how Better Bets constructs player form and its results:**

Form is a measure of how well a player is performing in a game. Player performance is observed at time $$t$$ for which its estimation depends on several variables that we believe to be important contributing factors (discussed below). We measure his average performance for his lifetime and observe the deviation in this game $$t$$ from the trend. If he is doing beter than usual , he will be deviating upwards from the average. This 'deviation' is measured in the following ratio:

$$\text{Form in game t}=\frac{\text{Performance in game t}}{\text{Average performance}}$$

.
\
Hopefully it is clear that when this number is greater than 1 then this player is doing well and thus his form is higher.  Observeve multiple games with a ratio greater than 1 and he is on a 'hot-streak' (and of course the opposite for less than 1).\
The problem here is how to estimate player performance and average player performance!

**Estimating player performance** \
The way we thought to do this was in two parts:
* 'Raw' performance
* 'Relative' performance.

Raw performance is simply a players raw statistics for a game- the stuff that the players see themselves. This includes kills, deatha and an implied kill-death-ratio (KDR). In particular, we took KDR and kills as the relevant statistics to use here.

Relative performance refers loosley to how the player is performing in the scope of his team only. Having a relative measure of performance allows us to look past featues of a game that might 'disguise' a player having an excellent performance despite his team performing poorly (and thus a low level of kills on his part) and vice-versa. 

Defining relative performance required some creation of metrics (measurement) on our part, these are:

$$\frac{\text{Kills}}{\text{Team}\text{Kills}}$$,               $$\frac{\text{Deaths}}{\text{Team}\text{Deaths}}$$, $$\frac{KDR}{\text{TeamScore}}$$

So finally, lets give this metric some functional form and define a players performance at game $$T$$:

$$log \Big[\frac{K^{2}_{T}}{D^{2}_{T}}\Big]+log\Big[\frac{\frac{K_{T}}{TK_{T}}.\frac{KDR_{T}}{log[(1+TS_{T})](TS_{T}-ES_{T})}}{\frac{D_{T}}{TD_{T}}}\Big]$$

It's a bit dense, so let's expand this
![N|Solid](http://i64.tinypic.com/99gsih.jpg)

That's better, but it is quite intimidating. Lets discuss what is going on here.\
This equation has two parts, the 'raw' and 'relative' measure of performance; these are the first and second terms in this equation, respectivly. Logs are taken on both terms such that they return a constrained range of values.

The **first** term is simply the product (multiplying of) of  kills:K and KDR:$$\frac{K}{D}$$- this reflects that a higher KDR is better, with more kills in a game (regardless of deaths) is also good. We can consider the first term an enhanced KDR with an emphasis on kills. 
.

The **second** term is a bit more complex. It's components were already mentioned above, but with $$\frac{KDR_{T}}{\text{TeamScore}}$$ having some alterations done to it.
Namely, these alterations were:
1. To muliply TeamScore by (TeamScore-EnemyScore). Doing this allowed a players performance to be discounted in very one sided matches where their oponents are more likely to be of lower skill than the team.
2. Taking the logs of this product. This bounded the range TeamScore could take, as a range of 16 could affect this ratio more than intended.
3. Add one to TeamScore. This is just a mathematical trick as to avoid problems when dealing with the above log.

The relative meausure is strictly increasing in kills and decreasing in deaths. It is also (interestingly) decreasing in team kills and increasing in team deaths. In this respect we see a player with many kills and few deaths as good, while respecting a player more who can maintain such kills and deaths despite his team doing worse. This term therefore is a measure of his contribution to the team.

**Recall**: This is not the complete picture. We want to estimate 'form' in game T- this is defined by the ratio of performance in game t (defined above) and a players average performance. Thankfully, at this stage, we have already done most of the work- getting this average is quite trivial. An average of performance is simple to take this performance measure for each game in this players history (exluding game T) and to take the average! This is given below, using summation notation:

![N|Solid](http://i68.tinypic.com/x54av.jpg)

An even more formidable equation...but let me talk you through it and you will see it is not so bad. Remember that we are simply taking the average of a players performance for all games excluding game T. In this respect, we can refer to the $$\sum$$ symbol- called 'Sigma' and the fraction $$\frac{1}{T-1}$$. What Sigma does is sum (add)- in particular, it is summing from the bottom part (0) to the top (T-1). If you treat a players first ever game as game 0 and the game right before game T as game T-1, then this Sigma this is summing the player performances across all of the games they have every played professionally, up until game T. The second term takes this sum and is dividing it by T-1. How do you find an average? You add everything up and divide by how many things there were: this is simply what we are doing- nothing scary.

Finally, the overall expression of **form** in its full extent- our true goal:

![N|Solid](http://i64.tinypic.com/bhhfmf.jpg)
