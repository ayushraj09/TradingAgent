# Paper Trading Flow (Alpaca) — FinRL

This document explains the live / paper trading flow implemented in `finrl/meta/paper_trading/alpaca.py` and the related data processing in `finrl/meta/data_processors/processor_alpaca.py`. It describes how data is fetched and cleaned, how the turbulence signal is calculated and used, how the agent state is constructed, and how model actions are converted into real orders.

---

## 1. High-level flow

1. Model is loaded into `PaperTradingAlpaca` (PPO agent from one of the supported libraries).
2. Each trading iteration calls `get_state()` to fetch the latest market data and build the agent state.
3. The agent predicts an action vector from the state.
4. Trading logic interprets the action vector and places orders via Alpaca's REST API.
5. If a turbulence risk flag is triggered, all positions are liquidated (risk-off).

---

## 2. Model loading

- The `PaperTradingAlpaca` constructor loads a trained agent depending on `drl_lib`:
  - `elegantrl`: constructs an `AgentPPO` then loads weights from `actor.pth`. Final action is produced through `agent.act` (a PyTorch module).
  - `stable_baselines3`: loads a `PPO` model with `PPO.load(cwd)`.
  - `rllib`: loads a Ray RLlib `PPOTrainer` checkpoint.

Note: In the repository code, `elegantrl` predictions are explicitly scaled (see §Action scaling). `stable_baselines3` predictions are obtained via `model.predict(state)[0]` (the original file does not scale SB3 outputs; some example wrappers in notebooks apply scaling to SB3 for parity).

---

## 3. Data pipeline (Alpaca -> DataFrame)

Data fetching and processing primarily happen in `AlpacaProcessor` (`processor_alpaca.py`). Two relevant entry points:

- `download_data(...)` — full historical download and reformat.
- `fetch_latest_data(ticker_list, time_interval, tech_indicator_list, limit=100)` — used in live/paper trading to obtain the most recent minute bars and technical indicators.

What `fetch_latest_data` does (summary):
- For each ticker in `ticker_list` it requests recent minute bars from Alpaca's `StockHistoricalDataClient`.
- It normalizes/renames timestamp/symbol columns, ensures `timestamp` is the index, and extracts columns.
- It builds a per-minute time index for the returned time range and forward-fills / back-fills missing values for each ticker.
- It calls `add_technical_indicator(...)` to compute indicators (uses `stockstats` / `StockDataFrame`).
- It creates arrays with:
  - `price_array` — close prices shaped [time, tickers] (the code returns latest row as `latest_price`).
  - `tech_array` — technical indicators flattened (concatenated per ticker).
  - `turbulence_array` — originally a turbulence column / or VIX value depending on path.

Resulting DataFrame columns (after cleaning and adding indicators):
- Core OHLCV columns per row: `timestamp`, `tic`, `open`, `high`, `low`, `close`, `volume`.
- Additional columns: technical indicator columns (examples: `macd`, `boll_ub`, `boll_lb`, `rsi_30`, `dx_30`, `close_30_sma`, `close_60_sma`, etc.).
- When `add_turbulence` is used offline, a `turbulence` column is added.
- For the live `fetch_latest_data` path, the code queries the `VIXY` instrument and returns its latest `close` value as `latest_turb` (used as real-time turbulence proxy).

Note: internal arrays returned to `get_state()` are `price` (latest close for each ticker), `tech` (latest technical indicator values flattened), and `turbulence` (currently the VIXY close value returned as `latest_turb`).

---

## 4. Turbulence — definition and calculation

There are two turbulence-related implementations in the repository:

1. Historical turbulence index (calculated in `calculate_turbulence`):
   - Build a pivoted close-price table indexed by timestamp and columns = tickers.
   - Convert to returns via `pct_change()`.
   - For each date after an initial `time_period` window (default `252`):
     - Use the previous `time_period` returns as `hist_price`.
     - Drop columns/tickers with too many NaNs.
     - Compute covariance matrix `cov_temp` of historical returns.
     - Compute `current_temp = (current_return - mean(hist))`.
     - Compute a Mahalanobis-like value: `temp = current_temp * pinv(cov_temp) * current_temp.T`.
     - Under some initial conditions the value is set to `0` to avoid outliers. Otherwise the Mahalanobis distance-like value becomes the turbulence index for that timestamp.
   - The output is a `DataFrame` of `{timestamp, turbulence}`.

   Interpretation: large Mahalanobis distance => current multivariate returns are far from historical distribution => high turbulence.

2. Live/latest turbulence used in `fetch_latest_data`:
   - The live fetch code retrieves `VIXY`'s latest `close` and returns `latest_turb` (a scalar array). `VIXY` is used as a volatility proxy in real-time rather than computing a covariance-based index on the fly.

---

## 5. How turbulence is used in paper trading (threshold / boolean)

- In `get_state()` (in `PaperTradingAlpaca`), after calling `fetch_latest_data` you receive `turbulence` as a numeric value (in live mode this comes from `VIXY` close).

- The code sets a boolean risk flag:
  ```python
  turbulence_bool = 1 if turbulence >= self.turbulence_thresh else 0
  ```
  So the comparison is `>=` the configured threshold. Default `turbulence_thresh` in the class constructor is `30`.

- If `turbulence_bool == 1` (i.e., turbulence >= threshold):
  - The trading loop switches to risk-off behavior and executes a full liquidation routine that iterates current Alpaca positions and submits orders to close them:
    - For each position: if `position.side == 'long'` submit a `sell` for the position qty; if `short` submit a `buy` to cover.
  - New buy orders are not placed while turbulence flag is set.

- If `turbulence_bool == 0` (normal market conditions), the agent's normal buy/sell decision logic runs.

---

## 6. Sigmoid transform and state normalization

- After determining the boolean flag, the raw `turbulence` is normalized for the agent state via `sigmoid_sign(ary, thresh)`:
  ```python
  def sigmoid_sign(ary, thresh):
      def sigmoid(x):
          return 1 / (1 + np.exp(-x * np.e)) - 0.5
      return sigmoid(ary / thresh) * thresh
  ```

- In `get_state()` the code then scales that value further:
  ```python
  turbulence = (self.sigmoid_sign(turbulence, self.turbulence_thresh) * 2**-5).astype(np.float32)
  ```
  So the sequence is: `sigmoid_sign` → multiply by 2^-5 (divide by 32) → cast to float32.

- Other scalings used in `get_state()`:
  - `tech` (technical indicators) is scaled by `2**-7` (divide by 128).
  - `price` is later scaled by `2**-6` (divide by 64) when building the state.
  - `cash` (amount) is scaled by `2**-12`.
  - `stocks` (holdings) are scaled by `2**-6`.

These scalings normalize features so the agent sees similar magnitudes across inputs.

---

## 7. State vector composition (`get_state()`)

The constructed `state` fed to the agent has the following concatenation order (in `PaperTradingAlpaca.get_state()`):

1. `amount` — scaled cash: `cash * 2**-12` (1 element)
2. `turbulence` — sigmoid-transformed & scaled turbulence (scalar)
3. `turbulence_bool` — 0 or 1 (scalar)
4. `price * scale` — scaled latest prices for each ticker (array length = `stock_dim`)
5. `stocks * scale` — holdings per ticker scaled (array length = `stock_dim`)
6. `stocks_cd` — cooldown / counters per ticker (array length = `stock_dim`)
7. `tech` — flattened technical indicators (length = `len(INDICATORS) * stock_dim`) (already scaled by `2**-7`)

In code:
```python
state = np.hstack((
    amount,
    turbulence,
    self.turbulence_bool,
    price * scale,
    self.stocks * scale,
    self.stocks_cd,
    tech,
)).astype(np.float32)
```

Notes:
- `stocks_cd` is a per-stock cooldown counter that increments each iteration and is zeroed when a trade is executed for that stock.
- Ensure your saved model's expected `state_dim` matches this structure. The example notebook shows there can be a mismatch between training-state-dim and paper-state-dim which must be reconciled (adapter shown in examples).

---

## 8. Agent actions and how they map to orders

- The agent produces an action vector, one value per ticker.

Current behavior in `trade()`:
- `elegantrl`:
  - Predicts via the actor: produces a tensor `a_tensor`.
  - Converted to NumPy and scaled: `action = (action * self.max_stock).astype(int)` — mapping continuous [-1, 1] to integer shares in `[-max_stock, max_stock]`.

- `stable_baselines3` (base file):
  - Calls `self.model.predict(state)[0]` and uses `action` as-is.
  - This may produce raw continuous values in `[-1, 1]` depending on the saved policy. Because `elegantrl` is explicitly scaled but SB3 is not in the base code, this can lead to inconsistent action units across libraries. In some examples (notebooks), SB3 output is multiplied by `max_stock` to align behavior.

- `rllib`:
  - `compute_single_action(state)` returns an action; treatment depends on the saved policy.

Action interpretation / execution logic (when `turbulence_bool == 0`):

- `min_action = 10` acts as an action-threshold.
  - Sell indices: `np.where(action < -min_action)[0]` — agent strongly negative → sell.
    - `sell_num_shares = min(self.stocks[index], -action[index])` → sell up to held shares (do not short beyond holdings in this routine).
    - Quantity `qty = abs(int(sell_num_shares))` — orders submitted via `submitOrder(qty, symbol, 'sell', respSO)`.
  - Buy indices: `np.where(action > min_action)[0]` — agent strongly positive → buy.
    - `buy_num_shares = min(tmp_cash // self.price[index], abs(int(action[index])))` → buy limited by available cash and the action magnitude.
    - Submit `submitOrder(qty, symbol, 'buy', respSO)` if qty > 0.

- If `turbulence_bool == 1` (high turbulence):
  - The code collects current positions via `self.alpaca.list_positions()` and for each position submits an opposite-side market order to close it (liquidate all holdings).

Order submission details (submitOrder):
- Calls `self.alpaca.submit_order(stock, qty, side, 'market', 'day')` and appends success/failure to a response list.
- If `qty <= 0` function appends `True` but does not place an order.

---

## 9. Cooldown (`stocks_cd`)

- `self.stocks_cd` increments each iteration: `self.stocks_cd += 1`.
- When a trade executes for a ticker, `self.stocks_cd[index] = 0` is assigned.
- This appears to act as a simple cooldown counter to prevent immediate repeated trades (used by the training environment as `stock_cd` concept). The code checks only `min_action` for trade threshold, but cooldown is included in the state so the agent can learn to use it.

---

## 10. Notes, caveats and recommended fixes

- State dimension mismatch: The training environment and paper trading `get_state()` may produce different `state_dim`. The example notebook solves this with a `StateAdapterPaperTrading` that maps the live state to the training state's layout. Always ensure `state_dim` passed into `PaperTradingAlpaca` matches the model's expected input shape.

- Action scaling inconsistency:
  - `elegantrl` path applies `action = (action * self.max_stock).astype(int)` before executing trades.
  - `stable_baselines3` in the base file does not multiply `raw_action` by `max_stock`. If your SB3 model outputs actions in `[-1, 1]`, you should scale it to shares like `action = (raw_action * self.max_stock).astype(int)` for consistent behavior.

- Turbulence source:
  - In historical/backtest mode the code can compute a Mahalanobis-distance-based turbulence index using a rolling covariance window.
  - In live mode `fetch_latest_data()` uses `VIXY`'s latest close as a real-time turbulence proxy. If `VIXY` data is unavailable, turbulence may be missing — the code must handle this case (the example notebook inspects presence of VIXY).

- Safety: Market orders are used (`'market'`, `'day'`). In real trading, consider limit orders, slippage, position sizing limits, and robust error handling.

---

## 11. Quick checklist for running paper trading safely

- Confirm `state_dim` and model input shapes match.
- Confirm consistent action scaling for the DRL library used.
- Validate `VIXY` (or turbulence source) availability and set a sensible `turbulence_thresh` for your risk tolerance.
- Test using paper account before going live.
- Add logging for predictions, orders, and account snapshots (the example notebook includes `log_prediction()` and CSV output).

---

## 12. Useful file references

- Paper trading main: `finrl/meta/paper_trading/alpaca.py`
- Data processing & turbulence: `finrl/meta/data_processors/processor_alpaca.py`
- Example notebook using an adapter & logging: `examples/FinRL_Simplified_Paper_Trading.ipynb`

---

If you want, I can:
- Add this file at a different path (root README vs package README),
- Inject automatic checks (asserts) into `get_state()` to detect state-dim mismatches,
- Add recommended SB3-scaling lines to `trade()` and create a small unit-test harness to verify action ranges.

Tell me which next step you prefer.