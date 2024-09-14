const express = require("express");
const cors = require("cors");
const mongoose = require("mongoose");
const dotenv = require("dotenv");
const bodyparser = require("body-parser");
const Forecast = require("./models/forcastModel");
const app = express();
app.use(cors({ origin: "*" }));
app.use(express.json());
app.use(bodyparser.json());

app.use(bodyparser.urlencoded({ extended: true }));



dotenv.config();





const connectDB = async () => {
    try {
      await mongoose.connect(process.env.MONGO_URL, {
        useNewUrlParser: true,
        useUnifiedTopology: true,
      });
      console.log("MongoDB Connection Succeeded.");
    } catch (err) {
      console.error("Error in DB connection: " + err);
    }
  };
  
  connectDB();
  app.get("/forecast", async (req,res)=>{
    try {
        const {district,date}=req.query;
        // console.log(district,date);
        const fetchdData= await Forecast.find(
            {
                District:district,
                Date:date
            },
            '-District -Date -Latitude -Longitude -_id'
        )
       console.log(fetchdData);
  const result=[
    {
      "Total Precipitation_Forecast":fetchdData[0].Total_Precipitation_Forecast,

      "Cape_Forecast": fetchdData[0].Cape_Forecast,
      "2m_dewpoint_temperature_Forecast":fetchdData[0].temperature_Forecast,
      "2m_temperature_Forecast":fetchdData[0].temperature_Forecast,
      "convective_inhibition_Forecast":fetchdData[0].convective_inhibition_Forecast,
      "convective_precipitation_Forecast":fetchdData[0].convective_precipitation_Forecast,
      "convective_rain_rate_Forecast":fetchdData[0].convective_rain_rate_Forecast,
      "evaporation_Forecast":fetchdData[0].evaporation_Forecast,
      "surface_pressure_Forecast":fetchdData[0].surface_pressure_Forecast,
      "total_totals_index_Forecast": fetchdData[0].total_totals_index_Forecast,
      "total_cloud_cover_Forecast":fetchdData[0].total_cloud_cover_Forecast,
      "k_index_Forecast": fetchdData[0].k_index_Forecast,
      "10m_v_component_of_wind_Forecast": fetchdData[0].v_component_of_wind_Forecast,
        "10m_u_component_of_wind_Forecast":fetchdData[0].u_component_of_wind_Forecast,
    
    }
 ]
 console.log(result);
        return res.json(result);
    } catch (error) {
        return res.send(error)
    }
  
});

const port = process.env.PORT || 4000;

app.listen(port, () => {
  console.log("listening on port " + port);
});