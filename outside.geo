//+
SetFactory("OpenCASCADE");
Circle(1) = {0, 0, 0, 10, 0, 2*Pi};
//+
Circle(2) = {0, 0, 0, 1, 0, 2*Pi};
//+
Curve Loop(1) = {1};
//+
Curve Loop(2) = {2};
//+
Plane Surface(1) = {1, 2};
//+
Curve Loop(3) = {1};
//+
Physical Curve(4) = {1};
//+
Physical Curve("Outer_boundary", 5) = {1};
//+
Physical Curve("Outer_boundary", 5) += {2};
//+
Physical Curve("Interface", 6) = {2};
//+
Physical Surface("Outside", 7) = {1};
//+
Physical Curve(4) -= {1};
