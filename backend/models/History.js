const mongoose = require("mongoose");

const historySchema = new mongoose.Schema(
  {
    disease: { type: String, required: true },
    confidence: { type: Number, required: true },
    treatment: { type: String },
    severity: {
      score: Number,
      level: String,
      note: String,
    },
    weather: {
      temperature_c: Number,
      humidity: Number,
      rain_mm: Number,
      wind_speed: Number,
      warning: String,
    },
    weather_note: String,
    imageName: String,
  },
  { timestamps: true }
);

module.exports = mongoose.model("History", historySchema);
