function [ h, p, a, ci, stats ] = PairedTTest( x, y, a )
%PAIREDTTEST
     a = a/100;
     [h, p, ci, stats] = ttest(x, y, 'Alpha', a);

end

