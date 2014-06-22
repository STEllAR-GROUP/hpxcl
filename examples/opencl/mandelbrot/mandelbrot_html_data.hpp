// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef MANDELBROT_HTML_DATA_HPP_
#define MANDELBROT_HTML_DATA_HPP_

    static const char mandelbrot_html[] = 
    "                                                                                           \n"
    "   <!DOCTYPE html>                                                                         \n"
    "   <html>                                                                                  \n"
    "     <head>                                                                                \n"
    "       <title>Mandelbrot</title>                                                           \n"
    "       <meta name=\"viewport\" content=\"initial-scale=1.0, user-scalable=no\">            \n"
    "       <meta charset=\"utf-8\">                                                            \n"
    "       <style>                                                                             \n"
    "         html, body, #map-canvas {                                                         \n"
    "           height: 100%;                                                                   \n"
    "           margin: 0px;                                                                    \n"
    "           padding: 0px                                                                    \n"
    "         }                                                                                 \n"
    "       </style>                                                                            \n"
    "       <script src=\"https://maps.googleapis.com/maps/api/js?v=3.exp\"></script>           \n"
    "       <script>                                                                            \n"
    "                                                                                           \n"
    "   var map;                                                                                \n"
    "   var moonTypeOptions = {                                                                 \n"
    "     getTileUrl: function(coord, zoom) {                                                   \n"
    "           return document.baseURI +                                                       \n"
    "             \"/\" + zoom + \"/\" + coord.x + \"/\" +                                      \n"
    "             coord.y + \".png\";                                                           \n"
    "     },                                                                                    \n"
    "     tileSize: new google.maps.Size(256, 256),                                             \n"
    "     maxZoom: 60,                                                                          \n"
    "     minZoom: 0,                                                                           \n"
    "     name: \"Mandelbrot\"                                                                  \n"
    "   };                                                                                      \n"
    "                                                                                           \n"
    "   var moonMapType = new google.maps.ImageMapType(moonTypeOptions);                        \n"
    "                                                                                           \n"
    "   function initialize() {                                                                 \n"
    "     var myLatlng = new google.maps.LatLng(0, 0);                                          \n"
    "     var mapOptions = {                                                                    \n"
    "       center: myLatlng,                                                                   \n"
    "       zoom: 1,                                                                            \n"
    "       streetViewControl: false,                                                           \n"
    "       mapTypeControlOptions: {                                                            \n"
    "         mapTypeIds: [\"mandelbrot\"]                                                      \n"
    "       }                                                                                   \n"
    "     };                                                                                    \n"
    "                                                                                           \n"
    "     map = new google.maps.Map(document.getElementById(\"map-canvas\"),                    \n"
    "         mapOptions);                                                                      \n"
    "     map.mapTypes.set('mandelbrot', moonMapType);                                          \n"
    "     map.setMapTypeId('mandelbrot');                                                       \n"
    "   }                                                                                       \n"
    "   google.maps.event.addDomListener(window, 'load', initialize);                           \n"
    "                                                                                           \n"
    "                                                                                           \n"
    "       </script>                                                                           \n"
    "     </head>                                                                               \n"
    "     <body>                                                                                \n"
    "       <div id=\"map-canvas\"></div>                                                       \n"
    "     </body>                                                                               \n"
    "   </html>                                                                                 \n"
    "                                                                                           \n";   


    static size_t mandelbrot_html_len = sizeof(mandelbrot_html);

#endif
