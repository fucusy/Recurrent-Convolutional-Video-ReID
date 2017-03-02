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
