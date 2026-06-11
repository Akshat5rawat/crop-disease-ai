const express = require("express");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");

const History = require("../models/History");

const router = express.Router();

// Vercel-compatible
const upload = multer({
  storage: multer.memoryStorage(),
});

router.post("/upload", upload.single("file"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({
      error: "No image file uploaded",
    });
  }

  const aiApiUrl =
    process.env.AI_API_URL || "http://127.0.1:5000";

  const formData = new FormData();

  formData.append(
    "file",
    req.file.buffer,
    req.file.originalname || "leaf.jpg"
  );

  if (req.body.lat) {
    formData.append("lat", req.body.lat);
  }

  if (req.body.lon) {
    formData.append("lon", req.body.lon);
  }

  try {
    const response = await axios.post(
      `${aiApiUrl}/predict`,
      formData,
      {
        headers: {
          ...formData.getHeaders(),
        },
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        timeout: 60000,
      }
    );

    const data = response.data;

    const entry = await History.create({
      disease: data.disease,
      confidence: data.confidence,
      treatment: data.treatment,
      severity: data.severity,
      weather: data.weather,
      weather_note: data.weather_note,
      imageName: req.file.originalname,
    });

    return res.json({
      ...data,
      historyId: entry._id,
    });
  } catch (error) {
    console.error("HF ERROR:", error.response?.data || error.message);

    return res.status(502).json({
      error:
        error.response?.data?.error ||
        error.response?.data?.details ||
        error.message,
    });
  }
});

module.exports = router;