# Support Vector Regression - SVR

- Support Vector Machines support **linear** and **non-linear** regression that we can refer to as SVR.
- Instead of trying to fir the largest possivle street between two classes while limiting margin violations, SVR tries to fit as many instances as possible on the street while limiting magain violations.
- The width of the street is controlled by a hyper parameter "Epsilon"

----------

- SVR performs linear regression in a higher "dimensional space".
- We can think of SVR as if each data point in the training represents it's own dimension. When you evaluate your kernel between a test point and a point in the training set, the resulting value gives you the coordinate of your test point in that dimension.
- The vector we get when we evaluate the point for all points in the training set, ![equation](http://www.sciweavers.org/upload/Tex2Img_1575307610/render.png) is the representation of the test poitns in the higher dimensional space.
- Once you have that vector you then use it to perform a linear regression.
