const mongoose = require('mongoose');

const forecastSchema = new mongoose.Schema({
  District: { type: String },
  Latitude: { type: Number },
  Longitude: { type: Number },
  Date: { type: String },
  Cape_Forecast: { type: Number },
  u_component_of_wind_Forecast: { type: Number },
  v_component_of_wind_Forecast: { type: Number },
  dewpoint_temperature_Forecast: { type: Number },
  temperature_Forecast: { type: Number },
  Total_Precipitation_Forecast: { type: Number },
  convective_inhibition_Forecast: { type: Number },
  convective_precipitation_Forecast: { type: Number },
  convective_rain_rate_Forecast: { type: Number },
  evaporation_Forecast: { type: Number },
  k_index_Forecast: { type: Number },
  surface_pressure_Forecast: { type: Number },
  total_cloud_cover_Forecast: { type: Number },
  total_totals_index_Forecast: { type: Number }
});

const Forecast = mongoose.model('forecast', forecastSchema);

module.exports = Forecast;
