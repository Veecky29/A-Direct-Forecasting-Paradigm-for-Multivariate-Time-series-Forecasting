# A-Direct-Forecasting-Paradigm-for-Multivariate-Time-series-Forecasting
A Direct Forecasting Paradigm for Multivariate Time-series Forecasting

<img width="864" height="368" alt="image" src="https://github.com/user-attachments/assets/9e8e17f7-164b-4956-845f-c263e1f59f0e" />

This is the flow chart of our proposed method, which contains four main stages. Follow this flow can realize a fluently reproduction.

The correct flow should be: Preprocessing, IF, reference, Filtering, TRF, reference, and testing.

*IMPORTANT: DO use vLLM for inference, it is very important in accelerating.

In the "preprocessing" file, FLOAT_FORMAT = "%.3f"  # this setting can be change based on the GPU memory usage, in our testing, 3, 4, and 5 do not show significant differences on results, but show differences on GPU memory usages.

If some unforeseen problems are encountered, you can try reverting to SWIFT version v3.9.
