
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "bhv_block.h"

#include "strategy.h"

#include "bhv_basic_tackle.h"

#include <rcsc/action/basic_actions.h>
#include <rcsc/action/body_go_to_point.h>
#include <rcsc/action/body_intercept.h>
#include <rcsc/action/neck_turn_to_ball_or_scan.h>
#include <rcsc/action/neck_turn_to_low_conf_teammate.h>

#include <rcsc/player/player_agent.h>
#include <rcsc/player/debug_client.h>
#include <rcsc/player/intercept_table.h>
#include <rcsc/player/say_message_builder.h>

#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>

#include "neck_offensive_intercept_neck.h"
#include "chain_action/field_analyzer.h"
#include <iostream>
using namespace rcsc;

bool Bhv_Block::execute(PlayerAgent * agent)
{
    const WorldModel & wm = agent->world();
    updateBlockCycle(wm);
    // updateBlockerUnum();
    Vector2D targetPos = blockPos[wm.self().unum()];
    Circle2D targetRegion = Circle2D(targetPos, 2);

    if(wm.countTeammatesIn(targetRegion, 1, false) >= 2)
        return false;

    if(!Body_GoToPoint2010(targetPos, 0.1, 100, 1.2, 1, false, 20).execute(agent))
    {
        Body_TurnToPoint(targetPos).execute(agent);
    }
    agent->debugClient().addCircle(targetPos, 1);
    agent->setNeckAction( new Neck_TurnToBallOrScan() );
    return true;
}

void Bhv_Block::updateBlockerUnum()
{
    int blockC = INT_MAX;
    for(int unum = 2; unum <= 11; unum++){
        if(blockCycle[unum] < blockC)
        {
            blockC = blockCycle[unum];
            blockerUnum = unum;
        }
    }
}

void Bhv_Block::updateBlockCycle(const WorldModel &wm)
{
    int oppMin = wm.interceptTable()->opponentReachCycle();
    Vector2D startDribble = wm.ball().inertiaPoint(oppMin);
    AngleDeg dribbleAngle = (ServerParam::i().ourTeamGoalPos() - startDribble).th();
    double dribbleSpeed = 0.8;
    Vector2D dribbleVel = Vector2D::polar2vector(dribbleSpeed, dribbleAngle);

    for(int unum = 1; unum <= 11; unum++)
    {
        Vector2D kickPos = startDribble;
        blockCycle[unum] = INT_MAX;
        const AbstractPlayerObject * tm = wm.ourPlayer(unum);
        if(tm == nullptr || tm->unum() != unum)
            continue;
        dlog.addText(Logger::BLOCK, "TM %d >>>>>>>>>>>>>>>>>>>>>>", unum);
        for(int cycle = oppMin + 1; cycle < oppMin + 30; cycle++)
        {
            kickPos += dribbleVel;
            int reachCycle = getBlockCycle(tm, kickPos, cycle);
            dlog.addText(Logger::BLOCK, "cycle:%d, pos:%.1f,%.1f, reach:%d", cycle, kickPos, reachCycle);
            if(reachCycle <= cycle)
            {
                dlog.addLine(Logger::BLOCK, tm->pos(), kickPos,255,0,0);
                blockCycle[unum] = cycle;
                blockPos[unum] = kickPos;
                break;
            }
        }
    }
}

int Bhv_Block::getBlockCycle(const AbstractPlayerObject *tm, Vector2D dribblePos, int cycle)
{
    Vector2D tmPos = tm->inertiaPoint(cycle);
    double kickableArea = tm->playerTypePtr()->kickableArea();
    if(tm->pos().dist(dribblePos) < kickableArea)
        return 0;
    if(tmPos.dist(dribblePos) < kickableArea)
        return 0;



    double dist = tmPos.dist(dribblePos) - kickableArea;
    int dashCycle = tm->playerTypePtr()->cyclesToReachDistance(dist);
    int turnCycle = 0;

    double diffAngle = ((dribblePos - tmPos).th() - tm->body()).abs();
    double speed = tm->vel().r();
    while(diffAngle > 15)
    {
        diffAngle -= tm->playerTypePtr()->effectiveTurn( ServerParam::i().maxMoment(), speed );
        speed *= tm->playerTypePtr()->playerDecay();
        turnCycle++;
    }
    return dashCycle + turnCycle;
    int reachCycle = FieldAnalyzer::predict_player_reach_cycle(tm,
                                                              dribblePos,
                                                              kickableArea,
                                                              0,
                                                              0,
                                                              0,
                                                              0,
                                                              false);
    return reachCycle;
}
