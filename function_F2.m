 function F = function_F2(X)
    F = zeros(size(X, 1), 1);
    for i = 1:size(X, 1)
        x1_val = X(i, 1);
        x2_val = X(i, 2);
        xi = 2.1 * x1_val - 0.1;
        delta = x2_val - xi;
        
        if delta >= 0.5
            F(i) = 1;
        elseif delta >= 0
            F(i) = 2 * delta;
        else
            center_x = 3/2;
            center_y = 1/2;
            r = sqrt((xi - center_x)^2 + (x2_val - center_y)^2);
            if r <= 0.25
                F(i) = (cos(4 * pi * r) + 1) / 2;
            else
                F(i) = 0;
            end
        end
    end
end