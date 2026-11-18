# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/radjahidayah-arch/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/radjahidayah-arch/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A forecaster for GTA VI's launch-day Twitch viewers, hour by hour. It reads a case (Twitch viewer counts up to the end of the release day's first hour) and forecasts hours 2-24 as five quantiles each, scored by mean pinball loss on the log scale.

## How it works

The model learns the launch-day shape from the case's comparables — past big game launches' hourly Twitch curves. It keeps only console-only launches shaped like GTA VI (PS5/Xbox, rolling midnight-local unlock), then regresses log(viewers_h / viewers_1) on log(viewers_1) per hour: bigger launches peak much less above hour 1. The median forecast is hour 1's print times that shape, and the quantiles use a per-hour log spread from the residuals, so the spread is honest about uncertainty.

Leave-one-out validation on the practice kit's comparables: pinball loss **0.160** vs the house persistence baseline's **0.344** (about 53% better; the house baseline's published practice score is ~0.387).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
```

---

Entered on [make.fun](https://make.fun/?ref=cqw52g77&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards
