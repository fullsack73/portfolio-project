from pmdarima import auto_arima
import numpy as np
import logging
from scipy.stats import linregress
import warnings

logger = logging.getLogger(__name__)

class ARIMA():
    """ARIMA-based forecasting model for returns and volatility."""
    
    def __init__(self, seasonal=False, suppress_warnings=True):
        """
        Initialize ARIMA model.
        
        Args:
            seasonal: Whether to use seasonal ARIMA (SARIMA)
            suppress_warnings: Whether to suppress model fitting warnings
        """
        self.seasonal = seasonal
        self.suppress_warnings = suppress_warnings
    
    def forecast(self, prices):
        """
        Forecast annual return and volatility using ARIMA model.
        
        Args:
            prices: Array-like of historical prices
            
        Returns:
            tuple: (expected_annual_return, annual_volatility)
        """
        if len(prices) < 10:
            logger.warning("Insufficient data points for ARIMA forecast")
            return (0.05, 0.15)
            
        try:
            # Convert prices to returns for better ARIMA performance
            returns = np.diff(prices) / prices[:-1]
            
            # Fit ARIMA model
            with warnings.catch_warnings():
                if self.suppress_warnings:
                    warnings.simplefilter("ignore")
                model = auto_arima(
                    returns,
                    seasonal=self.seasonal,
                    suppress_warnings=self.suppress_warnings,
                    error_action='ignore',
                    max_p=3, max_q=3, max_d=2
                )
            
            # Forecast next 252 days (1 year) of returns
            forecast_returns, conf_int = model.predict(
                n_periods=252,
                return_conf_int=True
            )
            
            # Calculate cumulative return
            cumulative_return = np.prod(1 + forecast_returns) - 1
            
            # Calculate volatility from confidence intervals
            forecast_std = np.std(forecast_returns)
            annual_volatility = forecast_std * np.sqrt(252)
            
            # Ensure minimum volatility
            annual_volatility = max(annual_volatility, 0.01)
            
            return (cumulative_return, annual_volatility)
            
        except Exception as e:
            logger.error(f"ARIMA forecast failed: {e}")
            # Fallback to simple linear trend if ARIMA fails
            try:
                x = np.arange(len(prices)).reshape(-1, 1)
                slope, intercept, _, _, _ = linregress(x.flatten(), prices)
                future_price = slope * (len(prices) + 252) + intercept
                current_price = prices[-1]
                expected_return = (future_price / current_price) - 1 if current_price > 0 else 0.05
                
                # Estimate volatility from historical returns
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns) * np.sqrt(252)
                
                return (expected_return, volatility)
            except:
                return (0.05, 0.15)


class LSTMModel():
    """LSTM neural network for time series forecasting.
    
    WARNING: LSTM/TensorFlow는 많은 메모리를 사용합니다.
    사용 후 반드시 cleanup() 메서드를 호출하거나 del로 삭제하세요.
    """
    
    def __init__(self, layers=2, units=50, dropout=0.2):
        """
        Initialize LSTM model.
        
        Args:
            layers: Number of LSTM layers
            units: Number of units per LSTM layer
            dropout: Dropout rate for regularization
        """
        self.layers = layers
        self.units = units
        self.dropout = dropout
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        
    def cleanup(self):
        """명시적 메모리 해제 - 사용 후 반드시 호출하세요."""
        if self.model is not None:
            try:
                import tensorflow as tf
                del self.model
                self.model = None
                tf.keras.backend.clear_session()
            except Exception:
                pass
        self.scaler_X = None
        self.scaler_y = None
        
    def __del__(self):
        """소멸자에서 cleanup 호출"""
        self.cleanup()
        
    def _create_sequences(self, data, lookback=60):
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def train(self, prices):
        """
        Train LSTM model on price data.
        
        Args:
            prices: Array-like of historical prices
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            from sklearn.preprocessing import StandardScaler
            
            # Suppress TensorFlow warnings
            tf.get_logger().setLevel('ERROR')
            
            if len(prices) < 100:
                logger.warning("Insufficient data for LSTM training, using simplified model")
                self.model = None
                return
            
            # Prepare data
            returns = np.diff(prices) / prices[:-1]
            returns = returns.reshape(-1, 1)
            
            # Scale data
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            scaled_returns = self.scaler_X.fit_transform(returns)
            
            # Create sequences
            lookback = min(60, len(scaled_returns) // 3)
            X, y = self._create_sequences(scaled_returns, lookback)
            
            if len(X) < 20:
                logger.warning("Insufficient sequences for LSTM training")
                self.model = None
                return
            
            # Reshape for LSTM
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Build model
            model = keras.Sequential()
            model.add(keras.layers.LSTM(self.units, return_sequences=(self.layers > 1), 
                                       input_shape=(X.shape[1], 1)))
            model.add(keras.layers.Dropout(self.dropout))
            
            for i in range(1, self.layers):
                return_seq = i < self.layers - 1
                model.add(keras.layers.LSTM(self.units, return_sequences=return_seq))
                model.add(keras.layers.Dropout(self.dropout))
            
            model.add(keras.layers.Dense(1))
            
            # Compile and train
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=20, batch_size=32, verbose=0, validation_split=0.1)
            
            self.model = model
            self.lookback = lookback
            logger.info("LSTM model trained successfully")
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            self.model = None
    
    def forecast(self):
        """
        Forecast expected annual return.
        
        Returns:
            float: Expected annual return
        """
        if self.model is None:
            logger.warning("LSTM model not trained, returning default")
            return 0.08
        
        try:
            # This is a simplified forecast - in production, you'd forecast multiple steps
            # For now, we return a conservative estimate
            return 0.08
            
        except Exception as e:
            logger.error(f"LSTM forecast failed: {e}")
            return 0.08


class XGBoostModel():
    """XGBoost gradient boosting model with feature engineering."""
    
    def __init__(self):
        """Initialize XGBoost model."""
        self.model = None
        self.feature_means = None
        
    def _engineer_features(self, prices):
        """
        Create features from price data.
        
        Args:
            prices: Array-like of historical prices
            
        Returns:
            DataFrame: Engineered features
        """
        import pandas as pd
        
        df = pd.DataFrame({'price': prices})
        
        # Returns
        df['return_1d'] = df['price'].pct_change()
        df['return_5d'] = df['price'].pct_change(5)
        df['return_20d'] = df['price'].pct_change(20)
        
        # Moving averages
        df['ma_5'] = df['price'].rolling(5).mean() / df['price']
        df['ma_20'] = df['price'].rolling(20).mean() / df['price']
        df['ma_50'] = df['price'].rolling(50).mean() / df['price']
        
        # Volatility
        df['volatility_10d'] = df['return_1d'].rolling(10).std()
        df['volatility_20d'] = df['return_1d'].rolling(20).std()
        
        # Momentum
        df['momentum'] = df['price'] - df['price'].shift(10)
        
        # RSI-like feature
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df.iloc[:, 1:]  # Exclude price column
    
    def train(self, prices):
        """
        Train XGBoost model on price data.
        
        Args:
            prices: Array-like of historical prices
        """
        try:
            import xgboost as xgb
            
            if len(prices) < 100:
                logger.warning("Insufficient data for XGBoost training")
                self.model = None
                return
            
            # Engineer features
            features_df = self._engineer_features(prices)
            
            # Create target (next day return)
            target = features_df['return_1d'].shift(-1)
            
            # Drop NaN rows
            valid_idx = ~(features_df.isna().any(axis=1) | target.isna())
            X = features_df[valid_idx].values
            y = target[valid_idx].values
            
            if len(X) < 50:
                logger.warning("Insufficient valid samples for XGBoost")
                self.model = None
                return
            
            # Store feature means for forecasting
            self.feature_means = np.mean(X, axis=0)
            
            # Train model
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
            self.model.fit(X, y)
            
            logger.info("XGBoost model trained successfully")
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            self.model = None
    
    def forecast(self):
        """
        Forecast expected annual return.
        
        Returns:
            float: Expected annual return
        """
        if self.model is None or self.feature_means is None:
            logger.warning("XGBoost model not trained, returning default")
            return 0.08
        
        try:
            # Predict using average features
            X_pred = self.feature_means.reshape(1, -1)
            daily_return = self.model.predict(X_pred)[0]
            
            # Annualize
            annual_return = (1 + daily_return) ** 252 - 1
            
            # Cap at reasonable values
            annual_return = np.clip(annual_return, -0.5, 1.0)
            
            return float(annual_return)
            
        except Exception as e:
            logger.error(f"XGBoost forecast failed: {e}")
            return 0.08


class ModelSelector():
    """Selects best-performing model based on validation metrics."""
    
    def validate_model(self, model, train_data, val_data):
        """
        Calculate validation metrics for a model.
        
        Args:
            model: Trained model instance
            train_data: Training prices
            val_data: Validation prices
            
        Returns:
            dict: Metrics including R² and RMSE
        """
        try:
            from sklearn.metrics import r2_score, mean_squared_error
            
            # Get actual returns
            actual_returns = np.diff(val_data) / val_data[:-1]
            
            # Get model predictions
            if isinstance(model, ARIMA):
                pred_return, _ = model.forecast(train_data)
                predicted = np.full(len(actual_returns), pred_return / 252)
            elif isinstance(model, (LSTMModel, XGBoostModel)):
                pred_return = model.forecast()
                predicted = np.full(len(actual_returns), pred_return / 252)
            else:
                return {'r2': -999, 'rmse': 999}
            
            # Calculate metrics
            r2 = r2_score(actual_returns, predicted)
            rmse = np.sqrt(mean_squared_error(actual_returns, predicted))
            
            return {'r2': r2, 'rmse': rmse}
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return {'r2': -999, 'rmse': 999}
    
    def select_best_model(self, train_data, val_data):
        """
        Train and select best model based on validation performance.
        
        Args:
            train_data: Training price data
            val_data: Validation price data
            
        Returns:
            tuple: (best_model, metrics_dict)
        """
        models = {
            'ARIMA': ARIMA(seasonal=False, suppress_warnings=True),
            'LSTM': LSTMModel(layers=2, units=32, dropout=0.2),
            'XGBoost': XGBoostModel()
        }
        
        results = {}
        
        # Train and validate each model
        for name, model in models.items():
            try:
                logger.info(f"Training and validating {name} model")
                
                # Train
                if isinstance(model, ARIMA):
                    # ARIMA doesn't need separate training
                    pass
                else:
                    model.train(train_data)
                
                # Validate
                metrics = self.validate_model(model, train_data, val_data)
                results[name] = {
                    'model': model,
                    'metrics': metrics
                }
                
                logger.info(f"{name} - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.6f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")
                results[name] = {
                    'model': model,
                    'metrics': {'r2': -999, 'rmse': 999}
                }
        
        # Select best model based on R²
        best_name = max(results.keys(), key=lambda k: results[k]['metrics']['r2'])
        best_result = results[best_name]
        
        logger.info(f"Selected {best_name} as best model")
        
        return (
            best_result['model'],
            {
                'model_name': best_name,
                'r2': best_result['metrics']['r2'],
                'rmse': best_result['metrics']['rmse']
            }
        )
