# On form: *estimation of a players upcoming performance.*

Estimation of the outcome of a game requires knowledge and insight into many variables. Understanding all the variables that aﬀect the final outcome of a game, means you will know for certain what will happen. If only such a task was so simple…
The crux of trying to bet on the outcome of a game (or anything in life with uncertainty), is that the number of variables that contribute to the final outcome of a series of actions is infinitely numerous, and thus infinitely complex to measure. The term ‘estimation’ comes from approximating reality, we do not know all these variables, and want to extract the most important ones to estimate what will happen. We believe that one of these important variables to estimate is ‘player form’.


**What is 'player form'?** \
In short, form is a measure of how well a player is doing, compared to how he normally does. People sometimes describe a player being on form for long periods of time as on a ‘hot-streak’, while conversely, describing a player who is not on form as ‘in a slump’.
More formally for a single game, we measure form as the percentage deviation from their mean performance.
We argue (and hope you can agree) that understanding when a player is (or isn’t) on form, should give us some more confidence as to who is going to win the game.


**Estimation of player form** \
One thing you may be thinking to yourself is:


*‘Why do this? I am immersed in the CS:GO community, I watch all the games, I’m on CS:GO Reddit every day, I know if a player is on form or not’.*

This may be true, but can you give ‘form’ a quantifiable value? The problem we are solving is being able to quantify form. Simply put, we are assigning a number on form, which enables us to code this as an input to a Machine Learning algorithm. Quantifying form, not only makes this tangible, but makes comparison of form between diﬀerent players trivial.

**So without futher ado, let us talk about how Better Bets constructs player form and its results:**

Form is a measure of how well a player is performing in a game. Player performance is observed at time   for which its estimation depends on several variables that we believe to be important contributing factors (discussed below). We measure his average performance for his lifetime, and observe the deviation in this game   from the trend. If he is doing better than usual, he will be deviating upwards from the average. This ‘deviation’ is measured in the following ratio. 

<img src="http://i65.tinypic.com/25r0dgz.jpg" alt="alt text" width="440" height="100">

Hopefully, it is clear that when this number is greater than 1, this player is doing well and thus, his form is higher. Observe multiple games with a ratio greater than 1, and he is on a ‘hot-streak’ (and conversely, the opposite for less than 1).

The problem here is how to estimate player performance and average player performance! 


**Estimating player performance** 
Note: *The actual form here is somewhat arbitrary. This can be considered a simplified version for explanations sake; in reality the model we use has many more variables than the ones we talk about here*.

The way we thought to model player performance was split into two parts:
* 'Raw' performance
* 'Relative' performance.

**Raw** performance is simply the raw statistics for a game, the stuﬀ that the players see themselves. This includes kills, deaths and an implied kill-death-ratio (KDR). In particular, we took KDR and kills as the relevant statistics to use here.

**Relative** performance refers loosely to how the player is performing in the scope of his team only. Having a relative measure of performance allows us to look past features of a game that might ‘disguise’ a player having an excellent performance despite his team performing poorly (and thus a low level of kills on his part) and vice-versa.

Defining relative performance required some creation of metrics (measurement) on our part, these are:


<img src="http://i63.tinypic.com/25i3xjq.jpg" alt="alt text" width="440" height="80">

So finally, lets give this metric some functional form and define a players performance at game **T**:

![N|Solid](http://i64.tinypic.com/99gsih.jpg)

That's better, but it is quite intimidating. Lets discuss what is going on here.

This equation has two parts, the 'raw' and 'relative' measure of performance; these are the first and second terms in this equation, respectivly. Logs are taken on both terms such that they return a constrained range of values.

The **first** term is simply the product of  kills and KDR- this reflects that a higher KDR is better, with more kills in a game (regardless of deaths) is also good. We can consider the first term an enhanced KDR with an emphasis on kills. 

The **second** termis a bit more complex. It's components were already mentioned above, but with $$\frac{KDR_{T}}{\text{TeamScore}}$$ having some alterations done to it.
Namely, these alterations were:
1. To muliply TeamScore by (TeamScore-EnemyScore). Doing this allowed a players performance to be discounted in very one sided matches where their oponents are more likely to be of lower skill than the team.
2. Taking the logs of this product. This bounded the range TeamScore could take, as a range of 16 could affect this ratio more than intended.
3. Add one to TeamScore. This is just a mathematical trick as to avoid problems when dealing with the above log.

The relative measure is strictly increasing in kills and decreasing in deaths. It is also (interestingly) decreasing in team kills and increasing in team deaths. In this respect, we see a player with many kills and few deaths as good, while respecting a player more who can maintain such kills and deaths despite his team doing worse. This term therefore is a measure of his contribution to the team.

Hopefully it is clear to see that this right hand side expression is indeed increasing in cases where we have more of a case to believe that a player is doing well/poorly in a single game.

**Recall**: This is not the complete picture. We want to estimate 'Form' in game T- this is defined by the ratio of performance in game t (defined above) and a players average performance. Thankfully, at this stage, we have already done most of the work- getting this average is quite trivial. An average of performance is simple to take this performance measure for each game in this players history (exluding game t) and to take the average! This is given (and expanded) below, using summation notation:

![N|Solid](http://i68.tinypic.com/x54av.jpg)

An even more formidable equation...but let me talk you through it and you will see it is not so bad. Remember that we are simply taking the average of a players performance for all games excluding game T. In this respect, we can refer to the large 'E' looking symbol symbol- called 'Sigma' and the fraction behind it. What Sigma does is sum (add)- in particular, it is summing from the bottom part (0) to the top (T-1). If you treat a players first ever game as game 0 and the game right before game T as game T-1, then this Sigma this is summing the player performances across all of the games they have every played professionally, up until game T. The second term is dividing everything by T-1..., but hold on- we are adding up the performances across 0 to T-1 games and then we are dividing this by T-1- isn't is just an expression for the avergage player performance? Yes it is, and this is why it is formulated like this. Not so scary after all.

Finally, the overall expression of **form** in its full extent- our true goal:

![N|Solid](http://i64.tinypic.com/bhhfmf.jpg)

#Machine Learning-RNN's and LSTM

The first half of this blog discussed what we wanted to know and an example of how we could estimate this. Namely- player form, and the equation defined just above. This works fine if you want to go back and have historical evidence of a players form: all you do is plug in the required variables of each game and you get out a number. In fact, you can do this for a player whole history and get a trend over time. 

How can one make money out of this? In particular, how are we able to estimate what a players form will be *before* the game has happened? This is the vital and having a good estimate *before* the fact is key to improving your insight as to what the outcome of a game should be and thus improving your payoffs the long-run. The solution to future estimation of form is where Machine Learning coming in; specifically an extension of Recurrent Neural Networks (RNN)-the Long-Term Short-Term (LSTM) model.

**RNN's:**
Before we can talk about LSTM, we need to cover the superset of models that they belong to


