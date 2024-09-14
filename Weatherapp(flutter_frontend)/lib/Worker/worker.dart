
import 'package:http/http.dart';
import 'dart:convert';
class worker{
  String? location;

  worker({this.location})
  {
    location = location;
  }
  String City="Dhaka";
  String? temp;
  String? humidity;
  String? air_speed;
  String? description;
  String? main;
  String? icon;
  Future<void> getData() async
  {
    try {
      Response response = await get(Uri.parse(
          "https://api.openweathermap.org/data/2.5/weather?q=$location,bd&appid=3c6fd24acf2351acae217890027cc334"));
      Map data = jsonDecode(response.body);
      print(data);
      Map tempData = data['main'];
      double getTemp = tempData['temp']-273.15;//c
      Map wind = data["wind"];//
      String getairSpeed = (wind["speed"]/.2777777).toString();
      String getHumidity = tempData["humidity"].toString();


      List weatherData = data['weather'];
      Map weatherMainData = weatherData[0];
      String getmainDes = weatherMainData['main'];
      String getDes = weatherMainData["description"];
      String geticon= weatherMainData["icon"].toString();
      print(icon);
      print(getTemp);
      print(weatherMainData['main']);


      // Assigning value
      temp = getTemp.toString();//C
      humidity = getHumidity;//%
      air_speed = getairSpeed;//Km/h
      description = getDes;
      main = getmainDes;
      icon = geticon;
    }catch(e)
    {
      temp = "Can't Find Data";
      humidity = "Can't Find Data";
      air_speed = "Can't Find Data";
      description = "Can't Find Data";
      main = "Can't Find Data";
      icon ="05d";
    }

  }

}

