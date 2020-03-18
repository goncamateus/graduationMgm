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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "role_center_forward.h"

#include "bhv_chain_action.h"
#include "bhv_basic_offensive_kick.h"
#include "bhv_basic_move.h"
#include "bhv_marlik_move.h"
#include "strategy.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/player/debug_client.h>

#include <rcsc/common/logger.h>

using namespace rcsc;

const std::string RoleCenterForward::NAME("CenterForward");

/*-------------------------------------------------------------------*/
/*!

 */
namespace
{
rcss::RegHolder role = SoccerRole::creators().autoReg(&RoleCenterForward::create,
                                                      RoleCenterForward::NAME);
}

/*-------------------------------------------------------------------*/
/*!

 */
bool RoleCenterForward::execute(PlayerAgent *agent)
{
    bool kickable = agent->world().self().isKickable();
    if (agent->world().existKickableTeammate() && agent->world().teammatesFromBall().front()->distFromBall() < agent->world().ball().distFromSelf())
    {
        kickable = false;
    }

    if (kickable)
    {
        if (agent->world().self().unum() == 11)
            std::cout << "CHAIN -> " << agent->world().time().cycle() << std::endl;
        doKick(agent);
    }
    else
    {
        if (agent->world().self().unum() == 11)
            std::cout << "MOVE -> " << agent->world().time().cycle() << std::endl;
        doMove(agent);
    }

    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool RoleCenterForward::execute(Agent *agent)
{
    bool kickable = agent->world().self().isKickable();
    if (agent->world().existKickableTeammate() && agent->world().teammatesFromBall().front()->distFromBall() < agent->world().ball().distFromSelf())
    {
        kickable = false;
    }

    if (kickable)
    {
        doKick(agent);
    }
    else
    {
        doMove(agent);
    }

    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
void RoleCenterForward::doKick(PlayerAgent *agent)
{
    if (Bhv_ChainAction().execute(agent))
    {
        dlog.addText(Logger::TEAM,
                     __FILE__ ": (execute) do chain action");
        agent->debugClient().addMessage("ChainAction");
        return;
    }

    Bhv_BasicOffensiveKick().execute(agent);
}

/*-------------------------------------------------------------------*/
/*!

 */
void RoleCenterForward::doMove(PlayerAgent *agent)
{
    //RoboCIn
    switch (Strategy::i().M_move_state)
    {
    case Strategy::MoveState::MS_BASIC_MOVE:
        Bhv_BasicMove().execute(agent);
        break;
    case Strategy::MoveState::MS_MARLIK_MOVE:
        Bhv_MarlikMove().execute(agent);
        break;
    default:
        Bhv_BasicMove().execute(agent);
        break;
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
void RoleCenterForward::doKick(Agent *agent)
{
    if (Bhv_ChainAction().execute(agent))
    {
        dlog.addText(Logger::TEAM,
                     __FILE__ ": (execute) do chain action");
        agent->debugClient().addMessage("ChainAction");
        agent->setLastActionStatus(true);
        return;
    }

    bool res = Bhv_BasicOffensiveKick().execute(agent);
    agent->setLastActionStatus(res);
}

/*-------------------------------------------------------------------*/
/*!

 */
void RoleCenterForward::doMove(Agent *agent)
{
    //RoboCIn
    switch (Strategy::i().M_move_state)
    {
    case Strategy::MoveState::MS_BASIC_MOVE:
        Bhv_BasicMove().execute(agent);
        break;
    case Strategy::MoveState::MS_MARLIK_MOVE:
        Bhv_MarlikMove().execute(agent);
        break;
    default:
        Bhv_BasicMove().execute(agent);
        break;
    }
}
