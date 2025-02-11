"""Modified GenCast configuration for India-specific weather forecasting."""

# 1. Define India's geographical bounds and climate zones
INDIA_BOUNDS = {
    'lat_min': 6.5,   # Southern tip
    'lat_max': 37.5,  # Northern boundary
    'lon_min': 68.0,  # Western boundary
    'lon_max': 97.5   # Eastern boundary
}

# India-specific climate zones
INDIA_CLIMATE_ZONES = [
    'tropical_monsoon',     # Most of peninsular India
    'tropical_savanna',     # Central India
    'hot_desert',          # Western India
    'humid_subtropical',    # Northern Plains
    'mountain_climate'      # Himalayan Region
]

# 2. Add India-specific variables
INDIA_SPECIFIC_VARS = (
    'monsoon_intensity',          # Monsoon strength indicator
    'convective_precipitation',   # Important for monsoon
    'soil_moisture',             # Critical for agricultural forecasting
    'boundary_layer_height'      # Important for pollution forecasting
)

# Modified target variables for India
TARGET_SURFACE_VARS = (
    '2m_temperature',
    'mean_sea_level_pressure',
    '10m_v_component_of_wind',
    '10m_u_component_of_wind',
    'total_precipitation_6hr',
    'sea_surface_temperature',
) + INDIA_SPECIFIC_VARS

# 3. Modify task configuration for India
INDIA_TASK = graphcast.TaskConfig(
    input_variables=(
        TARGET_SURFACE_NO_PRECIP_VARS
        + graphcast.TARGET_ATMOSPHERIC_VARS
        + graphcast.GENERATED_FORCING_VARS
        + graphcast.STATIC_VARS
        + INDIA_SPECIFIC_VARS
    ),
    target_variables=TARGET_SURFACE_VARS + graphcast.TARGET_ATMOSPHERIC_VARS,
    forcing_variables=graphcast.GENERATED_FORCING_VARS,
    pressure_levels=graphcast.PRESSURE_LEVELS_WEATHERBENCH_13,
    input_duration='6h',
    # Add geographical constraints
    region_bounds=INDIA_BOUNDS
)

# 4. India-specific sampler configuration
@chex.dataclass(frozen=True, eq=True)
class IndiaSamplerConfig:
    """Sampler configuration optimized for Indian weather patterns."""
    max_noise_level: float = 50.0  # Reduced for tropical climate
    min_noise_level: float = 0.02
    num_noise_levels: int = 30     # Increased for monsoon transitions
    rho: float = 9.0               # Adjusted for sharp weather changes
    # Stochastic sampling parameters
    stochastic_churn_rate: float = 3.5      # Increased for monsoon
    churn_min_noise_level: float = 0.4
    churn_max_noise_level: float = float('inf')
    noise_level_inflation_factor: float = 1.02
    # India-specific parameters
    monsoon_adjustment_factor: float = 1.2   # Extra weight during monsoon
    seasonal_scaling: float = 1.1           # Seasonal variation handling

# 5. India-specific model modifications
class IndiaGenCast(GenCast):
    """GenCast modified for India-specific weather forecasting."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.region_bounds = INDIA_BOUNDS
        self.climate_zones = INDIA_CLIMATE_ZONES
        
    def _apply_regional_mask(self, data: xarray.Dataset) -> xarray.Dataset:
        """Apply geographical mask for India region."""
        mask = (
            (data.lat >= self.region_bounds['lat_min']) &
            (data.lat <= self.region_bounds['lat_max']) &
            (data.lon >= self.region_bounds['lon_min']) &
            (data.lon <= self.region_bounds['lon_max'])
        )
        return data.where(mask, drop=True)
    
    def _get_season(self, data: xarray.Dataset) -> str:
        """Determine Indian season for the given data."""
        month = data.time.dt.month
        if month.isin([6, 7, 8, 9]):
            return 'monsoon'
        elif month.isin([10, 11]):
            return 'post_monsoon'
        elif month.isin([12, 1, 2]):
            return 'winter'
        else:
            return 'summer'
    
    def _adjust_for_season(self, loss: xarray.DataArray, data: xarray.Dataset) -> xarray.DataArray:
        """Apply seasonal adjustments to loss calculation."""
        season = self._get_season(data)
        season_weights = {
            'monsoon': 1.3,        # Higher weight during monsoon
            'post_monsoon': 1.1,
            'winter': 1.0,
            'summer': 1.2          # Higher weight during heat waves
        }
        return loss * season_weights[season]
    
    def loss(self, inputs: xarray.Dataset, targets: xarray.Dataset,
             forcings: Optional[xarray.Dataset] = None) -> predictor_base.LossAndDiagnostics:
        """Modified loss function with India-specific weights."""
        
        # Apply regional mask
        inputs = self._apply_regional_mask(inputs)
        targets = self._apply_regional_mask(targets)
        
        # Calculate base loss
        loss, diagnostics = super().loss(inputs, targets, forcings)
        
        # Apply India-specific variable weights
        india_weights = {
            'monsoon_intensity': 1.5,
            'convective_precipitation': 1.3,
            'soil_moisture': 1.2,
            'boundary_layer_height': 1.1
        }
        
        # Adjust loss for seasonal patterns
        loss = self._adjust_for_season(loss, inputs)
        
        # Apply monsoon-specific adjustments during monsoon season
        if self._get_season(inputs) == 'monsoon':
            loss *= self._sampler_config.monsoon_adjustment_factor
        
        return loss, diagnostics
    
    def _preconditioned_denoiser(self, inputs: xarray.Dataset, 
                                noisy_targets: xarray.Dataset,
                                noise_levels: xarray.DataArray,
                                forcings: Optional[xarray.Dataset] = None,
                                **kwargs) -> xarray.Dataset:
        """Enhanced denoiser with India-specific preconditioning."""
        # Apply regional constraints
        inputs = self._apply_regional_mask(inputs)
        noisy_targets = self._apply_regional_mask(noisy_targets)
        
        # Add seasonal information
        season = self._get_season(inputs)
        inputs['season'] = xarray.DataArray(
            data=[season],
            dims=('batch',)
        )
        
        return super()._preconditioned_denoiser(
            inputs, noisy_targets, noise_levels, forcings, **kwargs)

# 6. Custom evaluation metrics for India
def evaluate_india_metrics(predictions: xarray.Dataset, 
                         targets: xarray.Dataset) -> dict:
    """Calculate India-specific evaluation metrics."""
    metrics = {
        'monsoon_onset_error': calculate_monsoon_onset_error(predictions, targets),
        'heat_wave_prediction_accuracy': evaluate_heat_wave_prediction(predictions, targets),
        'extreme_rainfall_accuracy': evaluate_extreme_rainfall(predictions, targets),
        'regional_bias': calculate_regional_bias(predictions, targets, INDIA_BOUNDS)
    }
    return metrics
