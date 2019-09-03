// -*-c++-*-

/*
 *Copyright:

 Copyright (C) Hidehisa AKIYAMA

 This code is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3, or (at your option)
 any later version.

 This code is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this code; see the file COPYING.  If not, write to
 the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.

 *EndCopyright:
 */

/////////////////////////////////////////////////////////////////////

#ifndef BHV_BLOCK_H
#define BHV_BLOCK_H

#include <rcsc/geom/vector_2d.h>
#include <rcsc/player/soccer_action.h>
#include <rcsc/player/player_agent.h>


class Bhv_Block
    : public rcsc::SoccerBehavior {
public:
    Bhv_Block()
      { }

    bool execute( rcsc::PlayerAgent * agent );
private:
    void updateBlockerUnum();
    void updateBlockCycle(const rcsc::WorldModel & wm);
    int getBlockCycle(const rcsc::AbstractPlayerObject * tm, rcsc::Vector2D pos, int cycle);
    int blockCycle[12];
    rcsc::Vector2D blockPos[12];
    int blockerUnum = 0;

};

#endif
