# Automatic Market Making (AMM) under Constant Product Invariant

##  Introduction
A constant product invariant market allows trading a pair of asset after which the product of the reserve balance remain constant. Denotes $x$ and $y$ as the reserve balance in the pool, we can express it as `$x \cdot y = k$`. 

- When we exchange `$\Delta y$` for `$\Delta x$`,  the amount of `$x$` you will get is determined by`$(x - \Delta x)(y + \Delta y) = k$`. 
- Using $x$ as numeraire, the price of $y$ can be written as `$S = -\frac{dx}{dy} = \frac{k}{y^2} = \frac{x}{y}$`.  
- We can also rewrite`$x$` and`$y$` as  `$x = \sqrt{kS}$` and  `$y = \sqrt{\frac{k}{S}}$`.

## LP Portfolio Return - No Fee
Assuming no fee is collected for exchanging with LPs, the portfolio value of the LPs can be written as $V(t) = x(t) + y(t)S_u(t) = 2x(t) = 2\sqrt{kS_u(t)}$ and the simple buy and holder (HOLDers)'s portfolio value can be written as $\tilde{V}(t) = x(0) + y(0)S_u(t) = (1 + R(0, t))S(0)y(0)$.

The portfolio return for HOLDers can be written as $\tilde{V}(t)/\tilde{V}(0)  = \frac{1 + R(0, t)}{2}$ and $V(t)/V(0)  = \frac{1}{2} \sqrt{R_(0, t)}$ for LP, where $R_(0, t) = S(t)/S(0)$ as we can see from the plot the LP portfolio has negative convexity and only equals to the buy and hold portfolio when price is not change and underperform regardless which directly price moves. 


## LP Portfolio with Fees
It doesn't make sense to be a LP given the analysis above, but Martin Tassy and David White argues in [this paper](https://math.dartmouth.edu/~mtassy/articles/AMM_returns.pdf) that when taking the fees into the consideration *LPs on Uniswap will do better than HOLDers over time, even when the only incoming trades are arbs*.

To start with, we assume there exists an external which has access to infinite liquidity and trade against the AMM every time there is an arbitrage opportunity. 

### AMM with trading fee $1 - \gamma$

- The percentage fee LPs charge for trading on DEX is $1 - \gamma$. 
- A transaction in this market for trading $\Delta y$ for $\Delta x$ must satisfy: $$(x - \Delta x)(y + \gamma \Delta y) = k$$
- After the transaction the reserve will be updated following: $x \to x + \Delta x$, $y \to y + \Delta y$  and $k \to (x - \Delta x)(y + \Delta y)$. With the trading fee, the trading fee the reserve will grow.

### No Arbitrage Condition

Assuming that no-arbitrage condition is satisfied, we can show that the price from AMM can only deviates from the CEX price by at most a factor of $\gamma$. Denote $S_u$ as the DEX price, $S_m$ as the price from CEX and $S_a$ as abitrager's price.

- If we want to buy $\Delta y$, we have $(x + \gamma\Delta x)(y - \Delta y) = k$, thus $\frac{\gamma \Delta x}{\Delta y} = S_u$ and $p_a^{buy} = \gamma^{-1}S_u$
- If we want to sell $\Delta y$ , we have $ (y + \gamma\Delta y)(x - \Delta x) = k$, thus $\frac{\Delta x}{\gamma \Delta y} = S_u$ and $p_a^{sell} = \gamma S_u$
- The no-arbitrage condition is $S_a^{sell} \leq S_m \leq S_a^{buy}$, which leads to $\gamma S_m \leq S_u \leq \gamma^{-1} S_m$ and $\gamma S_u \leq S_m \leq \gamma^{-1} S_u$



https://arxiv.org/pdf/2106.14404.pdf

https://arxiv.org/pdf/1911.03380.pdf





 
