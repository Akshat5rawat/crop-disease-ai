const express = require("express");
const History = require("../models/History");

const router = express.Router();

router.get("/history", async (req, res, next) => {
  try {
    const limit = Number(req.query.limit || 20);
    const data = await History.find().sort({ createdAt: -1 }).limit(limit);
    return res.json(data);
  } catch (error) {
    return next(error);
  }
});

module.exports = router;
