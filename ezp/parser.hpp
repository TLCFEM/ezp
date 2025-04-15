/*******************************************************************************
 * Copyright (C) 2025 Theodore Chang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/

#ifndef PARSER_HPP
#define PARSER_HPP

#include <iterator>
#include <sstream>
#include <vector>

namespace ezp {
    namespace detail {
        struct cli_arg {
            std::string token;
            int value;
        };

        inline auto parse(const std::string& command) {
            std::istringstream iss(command);
            auto current = std::istream_iterator<std::string>(iss);
            const auto end = std::istream_iterator<std::string>();

            std::vector<cli_arg> args;

            while(current != end) {
                cli_arg arg{*current++, 0};
                if(!arg.token.starts_with("--")) continue;
                if(current == end) break;
                try {
                    arg.value = std::stoi(*current);
                    args.push_back(arg);
                    ++current;
                }
                catch(...) {
                }
            }

            return args;
        }
    } // namespace detail
} // namespace ezp

#endif

//! @}
