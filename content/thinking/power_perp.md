---
title: "Notes On Power Perpetual Contract"
date: 2022-01-18T02:50:16-04:00
categories: ["Math"]
tags: ["Blockchain"]
slug: power_perp
draft: false



---

# Background
Opyn just launched a power perpetual product for ETH named Squeeth (ETH^2) to provide option-like exposure with no-strike and no expiration dates on DEX. In this article we will go over the pricing formula for power contract, perpetual contract and then derived the power perpetual contract pricing using the traditional option pricing theory.


**Reference**

- [Power Perpetuals](https://www.paradigm.xyz/2021/08/power-perpetuals/)
- [Everlasting Option](https://www.paradigm.xyz/papers/everlasting_options.pdf)

# Pricing the Power Contract

Assuming `$S(t)$` follows GBM under risk-neutral measure,

$$
dS_t = S_t(rdt + \sigma dW_t)
$$

where `$S_t$` is the asset price, `$r$` is the risk-free rates and `$\sigma$` is the volatility. 

Let `$V_t = S_t^p$ and $f(t, S_t) = \ln V_t$`. According to Ito’s lemma we have,

`$$
df = f_t dt+ f_s dS + \frac{1}{2}f_{ss}dS^2
$$`

Given `$f_s = pS_t^{-1}$ and $f_{ss} = -p S_t^{-2}$`, we have

$$
df = pS_t^{-1} dS - \frac{1}{2}pS_t^{-2}dS^2 \\df = p(rdt + \sigma dW_t) - \frac{1}{2}p\sigma^2dt \\ df = (pr -  \frac{1}{2}p\sigma^2) dt + p \sigma dW_t
$$

Derivative pricing follows martingale under risk-neutral measure, thus we have
$$
V_t = E_t(e^{-r(T-t)}V_T) = E_t( S_t^pe^{(r(p-1) - \frac{1}{2}p\sigma^2)(T-t)+ p\sigma W_{T-t}})
$$

Thus,
$$
V_t = S_t^pe^{(r(p-1) + \frac{1}{2}\sigma^2p(p- 1))(T-t)}
$$

# Pricing Perpetual Contract

The creation of perpetual contract is to solve the inefficiency of rolling an option or future position as the spread is pretty wide and liquidity are wildly spread between maturity and strikes.

### Mechanism

- Market Price (`$P_M$`): the price the market is currently trading
- Payoff (`$P$`): the payoff at the maturity of the contract
- Funding Fee: `$\frac{P_M - P}{q}$`, `$q$` is number of payment per payment period (for example, 24 hourly payments per one-day funding period)

### Pricing and Replication

Assuming we are holding a portfolio of derivatives with maturity set `$\{t_i\}$`, the portfolio can be written as,
$$
V = \sum_{i = 0}^{+\infty}(1 - x)^ix P(t_i)
$$

when the front contract roll-off, the next contract will have `$x$` weight on the remaining portfolio and `$P(t_i)$` is the pricing of the derivative with matruity of `$t_i$`.

Now we see how we can find `$x$` to replicate the perpetual derivatives,

For perpetual derivative,

- Before funding payment at `$t_0$`, we sell `$x$` portion of our perpetual derivative for `$xP_M$`
- The remaining derivative position is `$1-x$`
- At funding time $t_0$, we pay funding `$(1-x)\frac{P_M - P}{q}$` for the remaining perpetual derivative

For replicated portfolio,

- At funding time `$t_0$`, our front contract expired and the payoff is `$xP$`
- The remaining replicated portfolio is `$1-x$`

To make the cash flow on perpetual the same as the replicated portfolio, we have,

$$
xP_M - (1-x)\frac{P_M - P}{q} = xP
$$

we get `$x = \frac{1}{1 + q}$`. Thus we can use the replicate portfolio to price the perpetual derivative here and derivative price

$$
V = \frac{1}{q}\sum_{i = 1}^{+\infty}(\frac{q}{1+q})^iP(t_i)
$$

### Future and Option

- If strike is set to 0, the everlasting option becomes perpetual future
- This framework can be used to price any funding-base perpetual derivative for which we can price the expiring equivalents.

  

# Pricing the Power Perpetual

Combing the power contract and perpetual contract pricing, we can get the pricing for power perpetual to be,

$$
V = S_t^p\frac{1}{q}\sum_{i = 1}^{+\infty}(\frac{q}{1+q})^i e^{Aif}
$$

where `$A = r(p-1) + \frac{1}{2}\sigma^2p(p- 1)$`, `$f$` is the funding frequency in years and $q$ is number of payment per payment period. Using some basic math, we have,

$$
(1 - \frac{q}{1+q}e^{Af})V =  S_t^p\frac{1}{1+q} e^{Af} \\ V =  S_t^p\frac{1}{(1+q)e^{-Af} -q} 
$$

Therefore, the funding rate can be written as,

$$
R= S_t^p(\frac{1}{(1+q)e^{-Af} -q} - 1)
$$

Remember for the summation to converge we need  `$\frac{q}{1+q}e^{Af} < 1$` to be satisfied.