const express = require("express");
const fs = require("fs");
const path = require("path");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");

const History = require("../models/History");

const router = express.Router();
const upload = multer({ dest: "uploads/" });

router.post("/upload", upload.single("file"), async (req, res, next) => {
  if (!req.file) {
    return res.status(400).json({ error: "No image file uploaded" });
  }

  const aiApiUrl = process.env.AI_API_URL || "http://127.0.0.1:5000";
  const localPath = path.resolve(req.file.path);

  const formData = new FormData();
  formData.append("file", fs.createReadStream(localPath), req.file.originalname || "leaf.jpg");

  if (req.body.lat) {
    formData.append("lat", req.body.lat);
  }
  if (req.body.lon) {
    formData.append("lon", req.body.lon);
  }

  try {
    const response = await axios.post(`${aiApiUrl}/predict`, formData, {
      headers: {
        ...formData.getHeaders(),
      },
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
      timeout: 30000,
    });

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
    const message =
      error.response?.data?.error ||
      error.response?.data?.details ||
      "AI API call failed. Ensure Flask service is running.";
    return res.status(502).json({ error: message });
  } finally {
    if (fs.existsSync(localPath)) {
      fs.unlinkSync(localPath);
    }
  }
});

module.exports = router;
