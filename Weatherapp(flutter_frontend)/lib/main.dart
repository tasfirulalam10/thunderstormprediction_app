//import 'dart:js';

import 'package:flutter/material.dart';
import 'package:weatherapp/Activity/Home.dart';
import 'package:weatherapp/Activity/Loading.dart';
//import 'package:thunderstorm_forecasting_page.dart';
void main() {
  runApp( MaterialApp(
    //home: Home() ,
    routes: {
      "/" : (context) => const Loading(),
      "/home" : (context) => const Home(),
      "/Loading" : (context) => const Loading(),
    },
  ));

}

