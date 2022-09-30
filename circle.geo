// Gmsh project created on Fri Sep 30 09:01:07 2022
SetFactory("OpenCASCADE");
//+
Circle(1) = {0, 0, 0, 1, 0, 2*Pi};
//+
Curve Loop(1) = {1};
//+
Curve Loop(2) = {1};
//+
Curve Loop(3) = {1};
//+
Curve Loop(4) = {1};
//+
Plane Surface(1) = {4};
//+
Physical Curve("outer", 5) = {1};
//+
Physical Surface("area", 6) = {1};
