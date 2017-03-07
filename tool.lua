--
-- Created by IntelliJ IDEA.
-- User: fucus
-- Date: 28/02/2017
-- Time: 7:43 PM
-- To change this template use File | Settings | File Templates.
--


function print_log(msg, level)
    print(string.format('%s[%s] %s', os.date('%Y-%m-%d %X'), level, msg))
end

function info(msg)
    print_log(msg, 'INFO')
end

function splitByComma(str)
    local res = {}
    for word in string.gmatch(str, '([^,\n]+)') do
        table.insert(res, word)
    end
    return res
end

function convertToType(output, c_type)
    if type(output) == 'table' then
        for i=1, #output do
            if c_type == 'double' then
                output[i] = output[i]:double()
            elseif c_type == 'cuda' then
                output[i] = output[i]:cuda()
            end
        end
    else
        if c_type == 'double' then
            output = output:double()
        elseif c_type == 'cuda' then
            output = output:cuda()
        end
    end
    return output

end

function convertToDouble(output)
    return convertToType(output, 'double')
end

function convertToCuda(output)
    return convertToType(output, 'cuda')
end
