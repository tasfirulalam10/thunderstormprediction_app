const mongoose = require('mongoose');
const Forecast = require("./models/forcastModel");
const dotenv = require("dotenv");





dotenv.config();

const mongoURI = process.env.MONGO_URL ;

// Connect to MongoDB
const connectDB = async () => {
  try {
    await mongoose.connect(mongoURI, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    console.log("MongoDB Connection Succeeded.");
  } catch (err) {
    console.error("Error in DB connection: " + err);
  }
};


// Function to update all District values
const updateDistricts = async () => {
  try {
    const forecasts = await Forecast.find();
    for (const forecast of forecasts) {
      if (forecast.District) {
        forecast.District = forecast.District.trim();
        await forecast.save();
      }
    }
    console.log('All District values have been updated.');
  } catch (error) {
    console.error('Error updating District values:', error);
  }
};

const runUpdate = async () => {
  await connectDB();
  await updateDistricts();
  mongoose.disconnect();
};

runUpdate();
