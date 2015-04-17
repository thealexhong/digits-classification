function displayTable(labels)
    colHeadings = arrayfun(@(x)sprintf('svm(%d)',x),0:9,'UniformOutput',false);
    format = repmat('%-9s',1,11);
    header = sprintf(format,'digit  |',colHeadings{:});
    fprintf('\n%s\n%s\n',header,repmat('-',size(header)));
    for idx = 1:numel(digits)
        fprintf('%-9s', [digits(idx) '      |']);
        fprintf('%-9d', sum(labels(:,:,idx)));
        fprintf('\n')
    end
end