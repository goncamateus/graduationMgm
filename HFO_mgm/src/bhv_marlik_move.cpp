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

#include "bhv_marlik_move.h"

#include "role_side_back.h"
#include "role_center_back.h"

#include "strategy.h"

#include "bhv_basic_tackle.h"

#include <rcsc/action/basic_actions.h>
#include <rcsc/action/body_go_to_point.h>
#include <rcsc/action/body_intercept.h>
#include <rcsc/action/neck_turn_to_ball_or_scan.h>
#include <rcsc/action/neck_turn_to_low_conf_teammate.h>
#include "neck_for_mark.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/player/debug_client.h>
#include <rcsc/player/intercept_table.h>

#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>

#include "neck_offensive_intercept_neck.h"

using namespace rcsc;


bool Bhv_MarlikMove::execute( PlayerAgent * agent )
{
    dlog.addText( Logger::TEAM,
                  __FILE__": Bhv_MarlikMove" );

    
    const WorldModel & wm = agent->world();

    // tackle
    if ( Bhv_BasicTackle( 0.8, 80.0 ).execute( agent ) && !wm.existKickableTeammate() )
    {
        return true;
    }

    Vector2D ball = wm.ball().pos();
    Vector2D me = wm.self().pos();
    
    const Vector2D target_point = Strategy::i().getPosition( wm.self().unum() );

    Vector2D homePos = target_point;

    int num = wm.self().unum();

    double dash_power = getDashPower( agent, homePos );

    const int self_min = wm.interceptTable()->selfReachCycle();
    const int mate_min = wm.interceptTable()->teammateReachCycle();
    const int opp_min = wm.interceptTable()->opponentReachCycle();



    Vector2D myInterceptPos = wm.ball().inertiaPoint( self_min );





{
    int goalCycles = 100;

    for( int z=1; z<15; z++ )
      if( wm.ball().inertiaPoint( z ).x < -52.5 && wm.ball().inertiaPoint( z ).absY() < 8.0 )
      {
         goalCycles = z;
         break;
      }

    if( (wm.self().unum() == 2 || wm.self().unum() == 3) && goalCycles != 100 &&
        mate_min >= goalCycles && me.x < -45.0 && !wm.existKickableTeammate() )
    {

      if( wm.ball().inertiaPoint( self_min ).x < -52.0 )
        Body_GoToPoint( wm.ball().inertiaPoint( goalCycles - 1 ),
                              0.5, ServerParam::i().maxDashPower() ).execute( agent );

      agent->setNeckAction( new Neck_OffensiveInterceptNeck() );
      return true;
    }
}


{
   if( me.x < -37.0 && opp_min < mate_min && opp_min > 2 &&
       homePos.x > -37.0 && wm.ourDefenseLineX() > me.x - 2.5 )
   {
        Body_GoToPoint( rcsc::Vector2D( me.x + 15.0, me.y ),
                        0.5, ServerParam::i().maxDashPower(), // maximum dash power
                        ServerParam::i().maxDashPower(),     // preferred dash speed
                        2,                                  // preferred reach cycle
                        true,                              // save recovery
                        5.0 ).execute( agent );

       if( wm.existKickableOpponent()
           && wm.ball().distFromSelf() < 12.0 )
             agent->setNeckAction( new Neck_TurnToBall() );
       else
             agent->setNeckAction( new Neck_TurnToBallOrScan() );
       return true;
   }

}

{
    if( crossMarkSideBack( agent ) )
      return true;

    if( crossMarkCenterBack( agent ) )
      return true;

    if( crossMarkDefensiveHalf( agent ) )
      return true;

    if( crossMarkOffensiveHalf( agent ) )
      return true;
}

{
  if( num == 6 && wm.self().stamina() > 3500.0 && ball.x > 33.0 )
  {
      Vector2D opp = wm.opponentsFromSelf().front()->pos();
      float oDist = wm.opponentsFromSelf().front()->pos().dist(homePos);

      if( oDist < 8.0 && opp.x > homePos.x )
        homePos = opp + Vector2D::polar2vector( 2.5 , (ball-opp).th() );

      if( homePos.x > 22.0 )
           homePos.x = 22.0;

  }

}

{
    if( wm.self().unum() == 6 && ball.x < 10.0 && ball.absY() > 12.0 && opp_min < mate_min && ball.x > -36.0)
    {
      Vector2D opp = wm.opponentsFromSelf().front()->pos();
      float oDist = wm.opponentsFromSelf().front()->pos().dist(homePos);

        if( oDist < 8.0 )
        {
          homePos = opp + Vector2D::polar2vector( 1.0 , (ball-opp).th() );
          homePos.x -= 1.0;
          isSoftMarking = true;
        }
    }
}

    Vector2D tempHomePos = homePos;

{
    const PlayerPtrCont & opps = wm.opponentsFromSelf();
    const PlayerObject * nearestOpp =
                ( opps.empty() ? static_cast< PlayerObject * >( 0 ) : opps.front() );
    const Vector2D opp = ( nearestOpp ? nearestOpp->pos() :
                                 Vector2D( -1000.0, 0.0 ) );

    double oppDist = opp.dist(homePos);

    if( ball.x > 34.0 && ball.absY() > 9.0 )
    {
      Vector2D markPoint = me + Vector2D::polar2vector(0.5, (ball-me).dir());

      if( wm.self().unum() == 11 && me.dist(homePos) < 10.0 )
      {
          if( opp.dist(markPoint) < 4.0 )
          {
            if( opp.x > me.x + 0.2 )
                homePos.x -= 2.5;
            else if( opp.x < me.x - 0.0 && homePos.x + 2.5 < wm.offsideLineX() )
                homePos.x += 2.5;
            else if( opp.dist(ball) < me.dist(ball) &&
                     opp.dist(ball) > me.dist(ball) - 2.0 )
                homePos += Vector2D::polar2vector(oppDist+2.0, (ball-me).dir());
          }
      }
      else if( wm.self().unum() == 9 || wm.self().unum() == 10 && me.dist(homePos) < 13.0)
      {
          if( opp.dist(markPoint) < 4.0 )
          {
            if( opp.x > me.x + 0.0 )
                homePos.x -= 3.0;
            else if( opp.x < me.x - 0.0 && homePos.x + 2.5 < wm.offsideLineX() )
                homePos.x += 2.5;
            else if( opp.dist(ball) < me.dist(ball) &&
                     opp.dist(ball) > me.dist(ball) - 3.0 )
                homePos += Vector2D::polar2vector(oppDist+2.0, (ball-me).dir());
          }
      }
      else if( wm.self().unum() == 7 || wm.self().unum() == 8 && me.dist(homePos) < 10.0)
      {
          if( opp.dist(markPoint) < 4.0 )
          {
            if( opp.x > me.x + 0.7 )
                homePos.x -= 3.0;
            else if( opp.x < me.x - 0.7 && homePos.x + 3.0 < wm.offsideLineX() )
                homePos.x += 3.0;
            else if( opp.dist(ball) < me.dist(ball) &&
                     opp.dist(ball) > me.dist(ball) - 3.0 )
                homePos += Vector2D::polar2vector(oppDist+2.0, (ball-me).dir());
          }
      }
    }

    if( num > 8 && mate_min < opp_min && mate_min < 5 && wm.self().stamina() > 3500.0 &&
        ball.x > -36.0 && wm.offsideLineX() < 36.0 && me.dist(homePos) < 6.0 )
    {
      Vector2D backPoint = me + Vector2D::polar2vector(2.0, (ball-me).dir());

      if( opp.dist(backPoint) < 2.0 )
      {
        if( homePos.x + 2.0 > wm.offsideLineX() )
           homePos.x -= 3.0;
        else
           homePos.x += 2.0;
      }
    }

    if( num > 5 && num < 9 && mate_min < opp_min && mate_min < 5 && wm.self().stamina() > 3000.0 &&
        ball.x > -36.0 && ball.x < 36.0 && me.dist(homePos) < 9.0 )
    {
      Vector2D markPoint = me + Vector2D::polar2vector(2.0, (ball-me).dir());

      if( opp.dist(markPoint) < 2.0 )
      {
         homePos += Vector2D::polar2vector(oppDist+2.0, (ball-me).dir());
      }
    }

    if( num > 5 && num < 9 && opp_min < mate_min && opp_min < 6 && wm.self().stamina() > 3000.0 &&
        ball.x > -36.0 && ball.x < 36.0 && me.dist(homePos) < 9.0 )
    {
      Vector2D markPoint = me + Vector2D::polar2vector(3.0, (ball-me).dir());

      if( opp.dist(markPoint) < 3.0 )
      {
         homePos += Vector2D::polar2vector(oppDist+1.5, (ball-me).dir());
      }
    }

}

{

    if( wm.self().unum() > 8 && mate_min < 10 )
    {
       if( ball.x > wm.offsideLineX() - 22.0 && std::fabs(ball.y - homePos.y) < 8.5 && ball.x < 36.0 )
       {
          if( wm.self().unum() == 11 )
          {
            if( ball.y < me.y )
                 homePos.y += 7.0;
            else
                 homePos.y -= 7.0;
          }
          else if ( wm.self().unum() == 9 )
                 homePos.y -= 7.0;
          else if ( wm.self().unum() == 10 )
                 homePos.y += 7.0;

          if( homePos.y > 33.0 )  homePos.y = 33.0;
          if( homePos.y < -33.0 ) homePos.y = -33.0;
       }

    }

}

{

    if( num == 2 || num == 3 )
       softMarkCenterBack( agent, homePos);

    if( num == 4 || num == 5 )
       softMarkSideBack2( agent, homePos);


    if( num < 6 && homePos.x < -35.0 && homePos.x > -37.0 && opp_min < 11 )
       homePos.x = -37.0;

    if( isSoftMarking && (num == 4 || num == 5) )
    {
      const PlayerPtrCont & opps = wm.opponentsFromSelf();
      const PlayerObject * nearest_opp = ( opps.empty() ? static_cast< PlayerObject * >( 0 ) : opps.front() );
      const Vector2D opp = ( nearest_opp ? nearest_opp->pos() : Vector2D( -1000.0, 0.0 ) );

      int oppConf = wm.opponentsFromSelf().front()->posCount();

      if( std::fabs(me.y - homePos.y) > 2.0 && me.x > homePos.x + 2.0 &&
          me.absY() < homePos.absY() )
      {
        homePos = rcsc::Vector2D(me.x-12.0, me.y);
      }

    }

}


{
    if( num < 6 && opp_min < mate_min && opp_min < 8 && wm.ourDefenseLineX() < homePos.x - 1.0 &&
        ball.x < 0.0 && ball.x > -35.0 )
    {
       homePos.x = std::max( -35.0, wm.ourDefenseLineX() );
    }

   if( num < 6 && opp_min < mate_min && opp_min < 15 && me.x > homePos.x + 2.5 && wm.ourDefenseLineX() < me.x - 0.5 &&
       ball.x < 0.0 && ball.x > -40.0 && std::fabs( me.absY() - homePos.absY() ) < 5.0 )
   {
       homePos.x = me.x - 10;
       homePos.y = me.y;
   }

   if( num < 9 && num > 5 && opp_min < mate_min && opp_min < 15 && me.x > homePos.x + 4 && wm.ourDefenseLineX() < me.x - 0.5 &&
       ball.x < 0.0 && ball.x > -40.0 && std::fabs( me.absY() - homePos.absY() ) < 7.5 )
   {
       homePos.x = me.x - 10;
       homePos.y = me.y;
   }


   if( ball.x < -30.0 && ball.x > -40.0 && ball.absY() < 20.0 && opp_min < mate_min && opp_min < 4 && homePos.x > me.x &&
       wm.ourDefenseLineX() < me.x - 1 )
   {
     if( num == 2 )
       homePos = rcsc::Vector2D(-46.0, -2.0);
     if( num == 3 )
       homePos = rcsc::Vector2D(-46.0, 2.0);
     if( num == 4 )
       homePos = rcsc::Vector2D(-46.0, -5.0);
     if( num == 5 )
       homePos = rcsc::Vector2D(-46.0, 5.0);
       
       
//     homePos.x = me.x;
     
     const PlayerPtrCont & opps = wm.opponentsFromSelf();
     const PlayerObject * nearestOpp =
                ( opps.empty() ? static_cast< PlayerObject * >( 0 ) : opps.front() );
     const Vector2D opp = ( nearestOpp ? nearestOpp->pos() :
                                 Vector2D( -1000.0, 0.0 ) );

     double oppDist = opp.dist(homePos);
    
     if( oppDist < 2.0 )
     {
//        homePos.x = opp.x - 0.5;
       homePos.y = opp.y;
     }
     
   }

    
}


{
  
  if( homePos.dist(ball) < 2.5 && wm.existKickableTeammate() && wm.teammatesFromBall().front()->distFromSelf() < 4.0 )
  {
    if( num <= 8 )
      homePos = homePos + rcsc::Vector2D::polar2vector( 4.0, (ball-homePos).dir() );
    if( num > 8 )
      homePos = homePos + rcsc::Vector2D::polar2vector( 5.0, (ball-homePos).dir() );
    
    
    if( homePos.x < wm.ourDefenseLineX() )
      homePos.x = wm.ourDefenseLineX();
    
    if( homePos.x > wm.offsideLineX() )
      homePos.x = wm.offsideLineX();
  }
  
}

//     if( num < 6 && homePos.x < -29.0 && homePos.x > -33.0 && opp_min < 12 )
//        homePos.x = -34.0;
//     if( num < 6 && homePos.x < -33.0 && homePos.x > -36.9 && opp_min < 12 )
//        homePos.x = -36.9;


///-------------------------------------------------------------------------------///

    double dist_thr = wm.ball().distFromSelf() * 0.1;
    if ( dist_thr < 1.0 ) dist_thr = 1.0;

      if( wm.self().unum() < 6 && isSoftMarking && homePos.x < -30.0 && homePos.absY() < 25.0 &&
          homePos.x > -39.0 && ball.dist(me) < 35.0 )
            dist_thr = 0.2;

    bool isUnmarking = false;

    if( tempHomePos.x != homePos.x || tempHomePos.y != homePos.y )
       isUnmarking = true;

    if( isUnmarking )
       dash_power = ServerParam::i().maxDashPower();

    AngleDeg bodyAngle = 0.0;

    if( wm.self().unum() > 8 &&
        wm.self().pos().x > wm.offsideLineX() - 0.2 && wm.self().pos().x < wm.offsideLineX() + 3.0 &&
        homePos.x > wm.offsideLineX() - 2.5 &&
        wm.self().body().abs() < 20.0 &&
        mate_min < opp_min && self_min > mate_min )
    {
         agent->doDash(-100, 0.0);
         if ( wm.existKickableOpponent()
              && wm.ball().distFromSelf() < 18.0 )
               agent->setNeckAction( new Neck_TurnToBall() );
         else
               agent->setNeckAction( new Neck_TurnToBallOrScan() );
         return true;
    }

    if( wm.self().unum() > 8 && ball.x < 30 && ball.x > -20 && me.dist(homePos) < 5 &&
        wm.self().pos().x < wm.offsideLineX() - 0.2 && // wm.self().pos().x > wm.offsideLineX() - 3.0 &&
        homePos.x > wm.offsideLineX() - 4.0 &&
        wm.self().body().abs() < 15.0 &&
        mate_min < opp_min && self_min > mate_min && wm.self().stamina() > 4500 )
    {
         agent->doDash(100, 0.0);
         agent->setNeckAction( new Neck_TurnToBallOrScan() );
         return true;
    }



    if( wm.self().unum() < 6 && homePos.x < -47.0 && fabs(homePos.y) < 6.0 )
    {
       if( !Body_GoToPoint( homePos, 0.8, ServerParam::i().maxDashPower() ).execute( agent ) )
       {
         AngleDeg bodyAngle = agent->world().ball().angleFromSelf();
        if( bodyAngle.degree() < 0.0 )
               bodyAngle -= 90.0;
        else
               bodyAngle += 90.0;

         Body_TurnToAngle( bodyAngle ).execute( agent );
         if ( wm.existKickableOpponent()
              || wm.ball().distFromSelf() < 18.0 )
               agent->setNeckAction( new Neck_TurnToBall() );
         else
               agent->setNeckAction( new Neck_TurnToBallOrScan() );
       }
     return true;
    }

    if( wm.self().unum() < 6 && opp_min < mate_min && opp_min < self_min &&
        (me.x > homePos.x + 1.0 || fabs(me.y - homePos.x) > 1.3) &&
        ball.x < homePos.x + 15.0 )
    {
         Body_GoToPoint( homePos,
                               0.8, ServerParam::i().maxDashPower() ).execute( agent );
         if ( wm.existKickableOpponent()
              && wm.ball().distFromSelf() < 18.0 )
               agent->setNeckAction( new Neck_TurnToBall() );
         else
               agent->setNeckAction( new Neck_TurnToBallOrScan() );
         return true;
    }

    if( ball.x > -36.0 && me.x < -40.0 && homePos.x > -40.0 )
    {
         Body_GoToPoint( Vector2D(me.x+10.0, me.y),
                               0.5, ServerParam::i().maxDashPower() ).execute( agent );
         if ( wm.existKickableOpponent()
              && wm.ball().distFromSelf() < 18.0 )
               agent->setNeckAction( new Neck_TurnToBall() );
         else
               agent->setNeckAction( new Neck_TurnToBallOrScan() );
         return true;
    }

    if( me.x > wm.offsideLineX() + 0.2 )
    {
         Body_GoToPoint( Vector2D(me.x-10.0, me.y),
                               0.5, ServerParam::i().maxDashPower() ).execute( agent );
         if ( wm.existKickableOpponent()
              && wm.ball().distFromSelf() < 18.0 )
               agent->setNeckAction( new Neck_TurnToBall() );
         else
               agent->setNeckAction( new Neck_TurnToBallOrScan() );
         return true;
    }

    if( num > 8 && mate_min < opp_min && ball.x > -15.0 &&
        homePos.x > wm.offsideLineX() - 3.0 && homePos.x < 36.0 && wm.self().stamina() > 4000.0 )
    {

       if( ! Body_GoToPoint( homePos,
                             0.5, ServerParam::i().maxDashPower(), // maximum dash power
                             ServerParam::i().maxDashPower(),     // preferred dash speed
                             2,                                  // preferred reach cycle
                             true,                              // save recovery
                             30.0 ).execute( agent ) )         // turn angle threshold
           Body_TurnToPoint( Vector2D(me.x + 10.0, me.y) ).execute( agent );


       if ( wm.existKickableOpponent()
            && wm.ball().distFromSelf() < 12.0 )
             agent->setNeckAction( new Neck_TurnToBall() );
       else
             agent->setNeckAction( new Neck_TurnToBallOrScan() );
       return true;

    }


    if ( ! Body_GoToPoint( homePos, dist_thr, dash_power
                           ).execute( agent ) )
    {
       if( wm.self().unum() < 6 && homePos.x < -30.0 && homePos.x > -38.0 && homePos.absY() < 20.0 &&
           opp_min < mate_min && opp_min < 10 && ball.dist(me) < 30.0 )
       {
         AngleDeg bodyAngle = agent->world().ball().angleFromSelf();
         if( bodyAngle.degree() < 0.0 )
               bodyAngle -= 90.0;
         else
               bodyAngle += 90.0;

         Body_TurnToAngle( bodyAngle ).execute( agent );
// 2011            Body_TurnToPoint( Vector2D(-49.0, 0.0) ).execute( agent );
       }
       else if( mate_min < opp_min &&
           homePos.x > wm.offsideLineX() - 15.0 &&
           wm.ball().pos().x > wm.offsideLineX() - 25.0 &&
//            wm.offsideLineX() < 30.0 &&
           (wm.self().unum() > 8 || (wm.self().unum() > 5 && wm.ball().pos().x > 30.0  )) )
       {
            Body_TurnToPoint( Vector2D(wm.self().pos().x + 20.0, wm.self().pos().y) ).execute( agent );
       }
       else if( wm.self().unum() < 6 && ball.x < wm.ourDefenseLineX() + 25.0 && mate_min > opp_min && opp_min < 10 )
       {
         AngleDeg bodyAngle = agent->world().ball().angleFromSelf();

         if( homePos.x < -34.0 && homePos.x > -37.0 )
             Body_TurnToPoint( rcsc::Vector2D(-37.0, 0.0) ).execute( agent );
//          else if ( bodyAngle.degree() < 0.0 )
//                 bodyAngle -= 90.0;
//          else
//                 bodyAngle += 90.0;
//             Body_TurnToAngle( bodyAngle ).execute( agent );
        else if( num == 4 || num == 5 )
             Body_TurnToPoint( rcsc::Vector2D(-50.0, 0.0) ).execute( agent );
        else
             Body_TurnToPoint( rcsc::Vector2D(-50.0, 0.0) ).execute( agent );

       }
       else if( espBodyTurns( agent, bodyAngle ) )
            Body_TurnToAngle( bodyAngle ).execute( agent );
       else
            Body_TurnToBall().execute( agent );

    }

    if( isSoftMarking || amIMarking )
    {
        agent->setNeckAction( new Neck_ForMark() );
        return true;
    }
    if( wm.existKickableOpponent()
        && wm.ball().distFromSelf() < 18.0 )
    {
        agent->setNeckAction( new Neck_TurnToBall() );
    }
    else
    {
        agent->setNeckAction( new Neck_TurnToBallOrScan() );
    }

    return true;
}

bool Bhv_MarlikMove::crossMarkCenterBack( PlayerAgent * agent )
{
  const WorldModel & wm = agent->world();

  Vector2D homePos = Strategy::i().getPosition( wm.self().unum() );
  Vector2D ball = wm.ball().pos();
  Vector2D me = wm.self().pos();
  Vector2D target = Vector2D(0.0,0.0);

  int num = wm.self().unum();
  double MarkDist = 1.2;

     if( ball.absY() < 15.0 )
         MarkDist = 1.0;

  if( ball.absY() < 9.0 )
       return false;

//   int myMin = wm.interceptTable()->selfReachCycle();
  int tmmMin = wm.interceptTable()->teammateReachCycle();
  int oppMin = wm.interceptTable()->opponentReachCycle();

  static bool isMarking = false;
  static int oppNo = -2;

  Vector2D S_opp;

  isMarking = false;

   if( ball.x > -20.0 || (ball.x > -36.0 && tmmMin < oppMin) )
   {
      isMarking = false;
      oppNo = -2;
   }

   if( oppNo != -2 )
   {
     const PlayerPtrCont::const_iterator end = wm.opponentsFromSelf().end();
     for ( PlayerPtrCont::const_iterator o = wm.opponentsFromSelf().begin();
           o != end;
           ++o )
     {
        if( (*o)->unum() == oppNo )
          if( (*o)->pos().x < homePos.x + 4.0 )
             S_opp = (*o)->pos();
          else
          {
            oppNo = -2;
            isMarking = false;
            break;
          }
     }
   }


  const PlayerPtrCont & opps = wm.opponentsFromSelf();
  const PlayerObject * nearestOpp = ( opps.empty() ? static_cast< PlayerObject * >( 0 ) : opps.front() );
  const Vector2D opp = ( nearestOpp ? nearestOpp->pos() : Vector2D( -1000.0, 0.0 ) );

  double oppDist = opp.dist(homePos);

  if( ( ( (num == 2 && ball.x < -37.0 && ball.y < homePos.y - 3.0) ||
          (num == 3 && ball.x < -37.0 && ball.y > homePos.y + 3.0)) &&
         oppDist < 2.5 ) || isMarking )
  {
//      if( oppNo == -2 )
         target = opp + Vector2D::polar2vector( MarkDist , (ball-opp).th() );
//      else if( S_opp.x != 0 )
//          target = S_opp + Vector2D::polar2vector( MarkDist , (ball-opp).th() );
//      else
//          return false;

     double dashPower = ServerParam::i().maxDashPower();
     double distThr = 0.4;

     if ( ! Body_GoToPoint( target, distThr, dashPower,
                                  -1.0,    // prefered dash speed. negative value means no preferred speed
                                  3,    // preferred reach cycle
                                  true,  // save_recovery
                                  15.0  // turn angle threshold
                                  ).execute( agent ) )
     {
         AngleDeg bodyAngle = agent->world().ball().angleFromSelf();
         if ( bodyAngle.degree() < 0.0 )
                bodyAngle -= 90.0;
         else
                bodyAngle += 90.0;

         Vector2D turnPoint = me;
         turnPoint += Vector2D::polar2vector( 10.0, bodyAngle );

         Body_TurnToPoint( turnPoint ).execute( agent );
     }

     isMarking = true;
     amIMarking = true;

     if( oppNo == -2 )
        oppNo == nearestOpp->unum();

    agent->setNeckAction( new Neck_TurnToBallOrScan() );
    return true;
  }

return false;
}

bool Bhv_MarlikMove::crossMarkSideBack( PlayerAgent * agent )
{
  const WorldModel & wm = agent->world();

  Vector2D homePos = Strategy::i().getPosition( wm.self().unum() );
  Vector2D ball = wm.ball().pos();
  Vector2D me = wm.self().pos();
  Vector2D target = Vector2D(0.0,0.0);

  int num = wm.self().unum();
  double MarkDist = 2.0;

     if( ball.dist(me) < 10.0 )
         MarkDist = 1.0;

     if( ball.absY() < 15.0 )
         MarkDist = 1.0;

  if( ball.absY() < 9.0 )
       return false;

//   int myMin = wm.interceptTable()->selfReachCycle();
  int tmmMin = wm.interceptTable()->teammateReachCycle();
  int oppMin = wm.interceptTable()->opponentReachCycle();

  static bool isMarking = false;
  static int oppNo = -2;

  Vector2D S_opp = Vector2D(0.0,0.0);

  isMarking = false;

   if( ball.x > -20.0 || (ball.x > -36.0 && tmmMin < oppMin) )
   {
      isMarking = false;
      oppNo = -2;
   }


   if( oppNo != -2 )
   {
     const PlayerPtrCont::const_iterator end = wm.opponentsFromSelf().end();
     for ( PlayerPtrCont::const_iterator o = wm.opponentsFromSelf().begin();
           o != end;
           ++o )
     {
        if( (*o)->unum() == oppNo )
          if( (*o)->pos().x < homePos.x + 5.0 )
             S_opp = (*o)->pos();
          else
          {
            oppNo = -2;
            isMarking = false;
            break;
          }
     }
   }


  const PlayerPtrCont & opps = wm.opponentsFromSelf();
  const PlayerObject * nearestOpp = ( opps.empty() ? static_cast< PlayerObject * >( 0 ) : opps.front() );
  const Vector2D opp = ( nearestOpp ? nearestOpp->pos() : Vector2D( -1000.0, 0.0 ) );

  double oppDist = opp.dist(homePos);



  if( ( ( (num == 4 && ball.x < -37.0 && ball.y > 4.0 && opp.y < me.y + 2.0 ) ||
          (num == 5 && ball.x < -37.0 && ball.y < -4.0 && opp.y > me.y - 2.0 ) ) &&
        oppDist < 10.0 && opp.x < -36.0 ) || isMarking )
  {
//      if( oppNo == -2 )
         target = opp + Vector2D::polar2vector( MarkDist , (ball-opp).th() );
//      else if( S_opp.x != 0 )
//          target = S_opp + Vector2D::polar2vector( MarkDist , (ball-opp).th() );
//      else
//          return false;

     double dashPower = ServerParam::i().maxDashPower();
     double distThr = 0.4;

     if ( ! Body_GoToPoint( target, distThr, dashPower,
                                  -1.0,    // prefered dash speed. negative value means no preferred speed
                                  3,    // preferred reach cycle
                                  true,  // save_recovery
                                  15.0  // turn angle threshold
                                  ).execute( agent ) )
     {
         AngleDeg bodyAngle = agent->world().ball().angleFromSelf();
         if ( bodyAngle.degree() < 0.0 )
                bodyAngle -= 90.0;
         else
                bodyAngle += 90.0;

         Vector2D turnPoint = me;
         turnPoint += Vector2D::polar2vector( 10.0, bodyAngle );

         Body_TurnToPoint( turnPoint ).execute( agent );
     }

     isMarking = true;
     amIMarking = true;

     if( oppNo == -2 )
        oppNo == nearestOpp->unum();

//      std::cout<<"\nCycle: "<<wm.time().cycle()<<" No "<<wm.self().unum()<<" MArrrrrrrrrrrrrrrrrrkiiiiiiiing "<<"\n";
     agent->setNeckAction( new Neck_TurnToBallOrScan() );
     return true;
  }

return false;
}

bool Bhv_MarlikMove::crossMarkDefensiveHalf( PlayerAgent * agent )
{
  const WorldModel & wm = agent->world();

  Vector2D homePos = Strategy::i().getPosition( wm.self().unum() );
  Vector2D ball = wm.ball().pos();
  Vector2D me = wm.self().pos();
  Vector2D target = Vector2D(0.0,0.0);

  int num = wm.self().unum();
  double MarkDist = 1.5;

     if( ball.absY() < 15.0 )
         MarkDist = 1.2;

  if( ball.absY() < 9.0 )
       return false;

//   int myMin = wm.interceptTable()->selfReachCycle();
  int tmmMin = wm.interceptTable()->teammateReachCycle();
  int oppMin = wm.interceptTable()->opponentReachCycle();

  static bool isMarking = false;
  static int oppNo = -2;

  isMarking = false;

  Vector2D S_opp = Vector2D(0.0,0.0);

   if( ball.x > -20.0 || (ball.x > -36.0 && tmmMin < oppMin) )
   {
      isMarking = false;
      oppNo = -2;
   }

   if( oppNo != -2 )
   {
     const PlayerPtrCont::const_iterator end = wm.opponentsFromSelf().end();
     for ( PlayerPtrCont::const_iterator o = wm.opponentsFromSelf().begin();
           o != end;
           ++o )
     {
        if( (*o)->unum() == oppNo )
          if( (*o)->pos().x < homePos.x + 5.0 )
             S_opp = (*o)->pos();
          else
          {
            oppNo = -2;
            isMarking = false;
            break;
          }
     }
   }

  const PlayerPtrCont & opps = wm.opponentsFromSelf();
  const PlayerObject * nearestOpp = ( opps.empty() ? static_cast< PlayerObject * >( 0 ) : opps.front() );
  const Vector2D opp = ( nearestOpp ? nearestOpp->pos() : Vector2D( -1000.0, 0.0 ) );

  double oppDist = opp.dist(homePos);

  if( ( num == 6 && ball.x < -37.0 && ball.absY() > 9.0 &&
        oppDist < 7.0 && opp.x > homePos.x - 4.0 ) || isMarking )
  {
//      if( oppNo == -2 )
         target = opp + Vector2D::polar2vector( MarkDist , (ball-opp).th() );
//      else if( S_opp.x != 0 )
//          target = S_opp + Vector2D::polar2vector( MarkDist , (ball-opp).th() );
//      else
//          return false;

     double dashPower = ServerParam::i().maxDashPower();
     double distThr = 0.4;

     if ( ! Body_GoToPoint( target, distThr, dashPower,
                                  -1.0,    // prefered dash speed. negative value means no preferred speed
                                  100,    // preferred reach cycle
                                  true,  // save_recovery
                                  20.0  // turn angle threshold
                                  ).execute( agent ) )
     {
         AngleDeg bodyAngle = agent->world().ball().angleFromSelf();
         if ( bodyAngle.degree() < 0.0 )
                bodyAngle -= 90.0;
         else
                bodyAngle += 90.0;

         Vector2D turnPoint = me;
         turnPoint += Vector2D::polar2vector( 10.0, bodyAngle );

         Body_TurnToPoint( turnPoint ).execute( agent );
     }

     isMarking = true;
     amIMarking = true;

     if( oppNo == -2 )
        oppNo == nearestOpp->unum();

    agent->setNeckAction( new Neck_TurnToBallOrScan() );
    return true;
  }

return false;
}

bool Bhv_MarlikMove::crossMarkOffensiveHalf( PlayerAgent * agent )
{
  const WorldModel & wm = agent->world();

  Vector2D homePos = Strategy::i().getPosition( wm.self().unum() );
  Vector2D ball = wm.ball().pos();
  Vector2D me = wm.self().pos();
  Vector2D target = Vector2D(0.0,0.0);

  int num = wm.self().unum();
  double MarkDist = 2.0;

     if( ball.absY() < 15.0 )
         MarkDist = 1.2;

  if( ball.absY() < 9.0 )
       return false;

//   int myMin = wm.interceptTable()->selfReachCycle();
  int tmmMin = wm.interceptTable()->teammateReachCycle();
  int oppMin = wm.interceptTable()->opponentReachCycle();

  static bool isMarking = false;
  static int oppNo = -2;

  isMarking = false;

  Vector2D S_opp = Vector2D(0.0,0.0);

   if( ball.x > -20.0 || (ball.x > -36.0 && tmmMin < oppMin) )
   {
      isMarking = false;
      oppNo = -2;
   }

   if( oppNo != -2 )
   {
     const PlayerPtrCont::const_iterator end = wm.opponentsFromSelf().end();
     for ( PlayerPtrCont::const_iterator o = wm.opponentsFromSelf().begin();
           o != end;
           ++o )
     {
        if( (*o)->unum() == oppNo )
          if( (*o)->pos().x < homePos.x + 5.0 )
             S_opp = (*o)->pos();
          else
          {
            oppNo = -2;
            isMarking = false;
            break;
          }
     }
   }

  const PlayerPtrCont & opps = wm.opponentsFromSelf();
  const PlayerObject * nearestOpp = ( opps.empty() ? static_cast< PlayerObject * >( 0 ) : opps.front() );
  const Vector2D opp = ( nearestOpp ? nearestOpp->pos() : Vector2D( -1000.0, 0.0 ) );

  double oppDist = opp.dist(homePos);

  if( ( ( (((num == 7 && ball.y > 9.0) || (num == 8 && ball.y < -9.0)) && fabs(opp.y - me.y) < 4.0 ) ||
          ((num == 8 && ball.y > 9.0) || (num == 7 && ball.y < -9.0)) ) &&
        ball.x < -37.0 && ball.absY() > 9.0 &&
        oppDist < 9.0 && opp.x > homePos.x - 2.0 && me.dist( homePos ) < 8.0 ) || isMarking )
  {
//      if( oppNo == -2 )
         target = opp + Vector2D::polar2vector( MarkDist , (ball-opp).th() );
//      else if( S_opp.x != 0 )
//          target = S_opp + Vector2D::polar2vector( MarkDist , (ball-opp).th() );
//      else
//          return false;

     double dashPower = ServerParam::i().maxDashPower();
     double distThr = 0.4;

     if ( ! Body_GoToPoint( target, distThr, dashPower,
                                  -1.0,    // prefered dash speed. negative value means no preferred speed
                                  3,    // preferred reach cycle
                                   true,  // save_recovery
                                  15.0  // turn angle threshold
                                  ).execute( agent ) )
     {
         AngleDeg bodyAngle = agent->world().ball().angleFromSelf();
         if ( bodyAngle.degree() < 0.0 )
                bodyAngle -= 90.0;
         else
                bodyAngle += 90.0;

         Vector2D turnPoint = me;
         turnPoint += Vector2D::polar2vector( 10.0, bodyAngle );

         Body_TurnToPoint( turnPoint ).execute( agent );
     }

     isMarking = true;
     amIMarking = true;

     if( oppNo == -2 )
        oppNo == nearestOpp->unum();


    agent->setNeckAction( new Neck_TurnToBallOrScan() );
    return true;
  }
return false;
}

void Bhv_MarlikMove::softMarkSideBack2( PlayerAgent * agent, Vector2D & homePos )
{
  const WorldModel & wm = agent->world();

  Vector2D ball = wm.ball().pos();
  Vector2D me = wm.self().pos();

  int self_min = wm.interceptTable()->selfReachCycle();
  int mate_min = wm.interceptTable()->teammateReachCycle();
  int opp_min = wm.interceptTable()->opponentReachCycle();

  const PlayerPtrCont & opps = wm.opponentsFromSelf();
  const PlayerObject * nearest_opp = ( opps.empty() ? static_cast< PlayerObject * >( 0 ) : opps.front() );
  Vector2D opp = ( nearest_opp ? nearest_opp->pos() : Vector2D( -1000.0, 0.0 ) );

  Vector2D oppFhome = Vector2D(0,0);

  int oppConf = wm.opponentsFromSelf().front()->posCount();
  int num = wm.self().unum();


  const PlayerPtrCont::const_iterator o2_end = wm.opponentsFromBall().end();
  for ( PlayerPtrCont::const_iterator o2 = wm.opponentsFromBall().begin();
        o2 != o2_end;
        ++o2 )
  {
         Vector2D oPos = (*o2)->pos() + (*o2)->vel();

         if( oPos.dist(opp) < 1.0 )
            continue;

         if( oPos.x < wm.ourDefenseLineX() + 10.0 && ( (num == 5 && oPos.y > opp.y+0.1) ||
                                                       (num == 4 && oPos.y < opp.y-0.1) ) )
         {
           opp = oPos;
         }

  }


  bool amInearest = true;

  double minDist = 100.0;
  
  const PlayerPtrCont::const_iterator o_end = wm.opponentsFromBall().end();
  for ( PlayerPtrCont::const_iterator o = wm.opponentsFromBall().begin();
        o != o_end;
        ++o )
  {
 
    Vector2D opos = (*o)->pos()+(*o)->vel();

    if( (*o)->posCount() > 10 )
         continue;

    if( opos.dist( homePos ) < minDist )
    {
         oppFhome = opos;
         minDist = opos.dist( homePos );
    }

  }

  const PlayerPtrCont::const_iterator t_end = wm.teammatesFromSelf().end();
  for ( PlayerPtrCont::const_iterator t = wm.teammatesFromSelf().begin();
        t != t_end;
        ++t )
  {
    Vector2D tpos = (*t)->pos()+(*t)->vel();

    if( (*t)->posCount() > 10 ) // round 2 IO2011: > 5
         continue;

    if( tpos.dist(opp) < me.dist(opp) )
    {
         amInearest = false;
         break;
    }
  }


  static bool marking = false;

  if( marking )
  {

    if( (wm.self().unum() == 4 && opp.y > homePos.y + 6.5) ||
        (wm.self().unum() == 5 && opp.y < homePos.y - 6.5) )
    {
       marking = false;
    }
    else if( agent->world().gameMode().type() != GameMode::PlayOn )
    {
       marking = false;
    }
    else if( nearest_opp->posCount() > 12 )
    {
       marking = false;
    }
    else if( ball.x < -35.0 )
    {
       marking = false;
    }
    else if( ball.x < -30.0 && std::fabs(ball.y-homePos.y) < 2.0 )
    {
       marking = false;
    }
    else if( mate_min < opp_min + 1 ) // 2011: opp_min + 3
    {
       marking = false;
    }
    else if( opp_min > 30 )
    {
       marking = false;
    }
    else if( ball.x > 10.0 )
    {
       marking = false;
    }
    else
    {

      if( (wm.self().unum() == 4 && opp.y < 0.0) ||
          (wm.self().unum() == 5 && opp.y > 0.0) )
        homePos.y = opp.y * 0.90; // 2011: * 0.975
      else
        homePos.y = opp.y * 1.15; // 2011: 1.1025

      marking = true;
      isSoftMarking = true;

      if( opp.x - 5.0 < homePos.x && std::fabs(ball.y-opp.y) > 10.0 )
           homePos.x = opp.x - 6.0;
      else if( opp.x - 4.5 < homePos.x && std::fabs(ball.y-opp.y) > 7.0 )
           homePos.x = opp.x - 5.0;
      else if( opp.x - 3.0 < homePos.x && std::fabs(ball.y-opp.y) > 5.0 )
           homePos.x = opp.x - 4.0;
      else if( opp.x - 2.0 < homePos.x ) // && std::fabs(ball.y-opp.y) > 2.5 )
           homePos.x = opp.x - 3.0;

   
      if( homePos.x < wm.ourDefenseLineX() - 4.0 ) 
         homePos.x = wm.ourDefenseLineX() - 4.0;
   

      if( homePos.x < -36.0 ) // was -37.0 before 2011
           homePos.x = -36.0;
      return;
    }



  }



  if( homePos.x > -36.5 && me.x < homePos.x + 4.0 && oppConf < 9 && opp.x < homePos.x + 10.0 &&
      opp.x > homePos.x - 3.0 && opp.x > wm.ourDefenseLineX() - 1.5 )
  {

    if( (wm.self().unum() == 4 && opp.y < homePos.y + 2.8) ||
        (wm.self().unum() == 5 && opp.y > homePos.y - 2.8) )
    {

//       std::cout<<"\nSoftMark for Side Executed! Cycle: "<<wm.time().cycle()<<" for "<<wm.self().unum()<<"\n";
      if( (wm.self().unum() == 4 && opp.y < 0.0) ||
          (wm.self().unum() == 5 && opp.y > 0.0) )
        homePos.y = opp.y * 0.9; // 2011: * 0.975
      else
        homePos.y = opp.y * 1.15; // 2011: * 0.1025 !!


      marking = true;

      if( opp.x - 5.0 < homePos.x && std::fabs(ball.y-opp.y) > 10.0 )
           homePos.x = opp.x - 6.0;
      else if( opp.x - 4.5 < homePos.x && std::fabs(ball.y-opp.y) > 7.0 )
           homePos.x = opp.x - 5.0;
      else if( opp.x - 3.0 < homePos.x && std::fabs(ball.y-opp.y) > 5.0 )
           homePos.x = opp.x - 4.0;
      else if( opp.x - 2.0 < homePos.x ) // && std::fabs(ball.y-opp.y) > 2.5 )
           homePos.x = opp.x - 3.0;

      if( homePos.x < wm.ourDefenseLineX() - 4.0 ) 
         homePos.x = wm.ourDefenseLineX() - 4.0;

      if( homePos.x < -36.0 )  // was -37.0 before 2011
           homePos.x = -36.0;

      isSoftMarking = true;
      return;
    }
    else
    {
      marking = false;
      isSoftMarking = false;
      return;
    }

  }
  else
  {
    marking = false;
    isSoftMarking = false;
    return;
  }
return;
}

void Bhv_MarlikMove::softMarkCenterBack( PlayerAgent * agent, Vector2D & homePos )
{
  const WorldModel & wm = agent->world();

  Vector2D ball = wm.ball().pos();
  Vector2D me = wm.self().pos();

  int self_min = wm.interceptTable()->selfReachCycle();
  int mate_min = wm.interceptTable()->teammateReachCycle();
  int opp_min = wm.interceptTable()->opponentReachCycle();

  const PlayerPtrCont & opps = wm.opponentsFromSelf();
  const PlayerObject * nearest_opp
      = ( opps.empty()
          ? static_cast< PlayerObject * >( 0 )
          : opps.front() );

  const Vector2D opp = ( nearest_opp
                                           ? nearest_opp->pos()
                                           : Vector2D( -1000.0, 0.0 ) );

  Vector2D oppFhome = Vector2D(0,0);
  
  double minDist = 100.0;
  
  bool amInearest = true;

  const PlayerPtrCont::const_iterator o_end = wm.opponentsFromBall().end();
  for( PlayerPtrCont::const_iterator o = wm.opponentsFromBall().begin();
        o != o_end;
        ++o )
  {

    Vector2D opos = (*o)->pos() + (*o)->vel();

    if( (*o)->posCount() > 10 )
         continue;

    if( opos.dist( homePos ) < minDist )
    {
         oppFhome = opos;
         minDist = opos.dist( homePos );
    }

  }

  const PlayerPtrCont::const_iterator t_end = wm.teammatesFromSelf().end();
  for ( PlayerPtrCont::const_iterator t = wm.teammatesFromSelf().begin();
        t != t_end;
        ++t )
  {

    Vector2D tpos = (*t)->pos() + (*t)->vel();

    if( (*t)->posCount() > 7 )
         continue;

    if( tpos.dist(opp) < me.dist(opp) && tpos.x < opp.x )
    {
         amInearest = false;
         break;
    }

  }


  static bool marking = false;
  static int oNo = -1;


  
  if( marking && oNo != -1 )
  {
    if( agent->world().gameMode().type() != GameMode::PlayOn )
    {
       marking = false;
       oNo = -1;
    }
    else if( nearest_opp->unum() != oNo )
    {
       marking = false;
       oNo = -1;
    }
    else if( nearest_opp->posCount() > 15 )
    {
       marking = false;
       oNo = -1;
    }
    else if( ball.x < -35.0 )
    {
       marking = false;
       oNo = -1;
    }
    else if( ball.x < -30.0 && std::fabs(ball.y-homePos.y) < 4.0 && std::fabs(me.y-homePos.y) > 3.0 )
    {
       marking = false;
       oNo = -1;
    }
    else if( mate_min < opp_min + 1 )
    {
       marking = false;
       oNo = -1;
    }
    else if( opp_min > 20 )
    {
       marking = false;
       oNo = -1;
    }
    else if( ball.x > 10.0 )
    {
       marking = false;
       oNo = -1;
    }
    else
    {
      homePos.y = opp.y;

      oNo = nearest_opp->unum();
      marking = true;
      isSoftMarking = true;

      if( opp.x - 5.0 < homePos.x && std::fabs(ball.y-opp.y) > 10.0 )
           homePos.x = opp.x - 6.0;
      else if( opp.x - 4.5 < homePos.x && std::fabs(ball.y-opp.y) > 7.0 )
           homePos.x = opp.x - 5.0;
      else if( opp.x - 3.0 < homePos.x && std::fabs(ball.y-opp.y) > 5.0 )
           homePos.x = opp.x - 4.0;
      else if( opp.x - 2.0 < homePos.x ) // && std::fabs(ball.y-opp.y) > 2.5 )
           homePos.x = opp.x - 3.0;

      if( homePos.x < wm.ourDefenseLineX() - 4.0 ) 
         homePos.x = wm.ourDefenseLineX() - 4.0;

      if( homePos.x < -36.0 )  // was -37.0 before 2011
           homePos.x = -36.0;

      return;
    }
    
  }

  float xDiff = 12.0;

  if( fabs(ball.x - homePos.x) < 15.0 )
     xDiff = 10.0;
  if( fabs(ball.x - homePos.x) < 10.0 )
     xDiff = 5.0;

  float yDiff = 7.0;

  if( homePos.x < -33.0 )
       yDiff = 4.5;
  if( homePos.x < -30.0 )
       yDiff = 5.0;
  if( homePos.x < -20.0 )
       yDiff = 8.0;


  if( homePos.x > -36.5 && me.x > homePos.x - 4.0 && me.x < homePos.x + 2.0 &&
      opp.x < homePos.x + xDiff && opp.x > homePos.x - 2.0 && opp.x > wm.ourDefenseLineX() - 1.5 )//&& amInearest )
  {
//       std::cout<<"\nSoftMark for Center Executed! Cycle: "<<wm.time().cycle()<<" for "<<wm.self().unum()<<"\n";
    if( ( wm.self().unum() == 2 && opp.y < homePos.y + (yDiff-2.0) && opp.y > homePos.y - yDiff ) ||
        ( wm.self().unum() == 3 && opp.y > homePos.y - (yDiff-2.0) && opp.y < homePos.y + yDiff ) )
    {
      homePos.y = opp.y;

      if( opp.x - 5.0 < homePos.x && std::fabs(ball.y-opp.y) > 10.0 )
           homePos.x = opp.x - 6.0;
      else if( opp.x - 4.5 < homePos.x && std::fabs(ball.y-opp.y) > 7.0 )
           homePos.x = opp.x - 5.0;
      else if( opp.x - 3.0 < homePos.x && std::fabs(ball.y-opp.y) > 5.0 )
           homePos.x = opp.x - 4.0;
      else if( opp.x - 2.0 < homePos.x ) // && std::fabs(ball.y-opp.y) > 2.5 )
           homePos.x = opp.x - 3.0;

      if( homePos.x < wm.ourDefenseLineX() - 4.0 ) 
         homePos.x = wm.ourDefenseLineX() - 4.0;

      if( homePos.x < -36.0 )  // was -37.0 before 2011
           homePos.x = -36.0;

      isSoftMarking = true;
    }
    else
      isSoftMarking = false;
  }
  else
      isSoftMarking = false;
}

bool Bhv_MarlikMove::espBodyTurns( const PlayerAgent * agent, AngleDeg & bodyAngle )
{
  const WorldModel & wm = agent->world();

  Vector2D ball = wm.ball().pos();
  Vector2D me = wm.self().pos();

  int num = wm.self().unum();


  if( (num == 2 || num == 3) && ball.absY() > 5.0 && me.x < -48.0 && sign(me.y) != sign(ball.y) )
  {
     bodyAngle = ( sign(ball.y) > 0.0 ? 90.0 : -90.0 );
     return true;
  }






return false;
}

double Bhv_MarlikMove::getDashPower( const PlayerAgent * agent,
                                    const Vector2D & target_point )
{
    static bool s_recover_mode = false;

    const WorldModel & wm = agent->world();

    Vector2D ball = wm.ball().pos();
    Vector2D me = wm.self().pos();

    /*--------------------------------------------------------*/
    // check recover

    int self_min = wm.interceptTable()->selfReachCycle();
    int mate_min = wm.interceptTable()->teammateReachCycle();
    int opp_min = wm.interceptTable()->opponentReachCycle();

    if ( wm.self().stamina() < ServerParam::i().staminaMax() * 0.5 )
    {
        s_recover_mode = true;
    }
    else if ( wm.self().stamina() > ServerParam::i().staminaMax() * 0.7 )
    {
        s_recover_mode = false;
    }

/// setting recover mode FALSE

    if( wm.self().unum() == 11 && ball.x > 33.0 && mate_min < opp_min &&
        mate_min < 10 && wm.self().stamina() > 2600 )
    {
        s_recover_mode = false;
    }

    if( (wm.self().unum() == 9 || wm.self().unum() == 10) && ball.x > 33.0 && mate_min < opp_min &&
        mate_min < 10 && wm.self().stamina() > 2600 )
    {
        s_recover_mode = false;
    }


    if( wm.self().unum() == 8 && ball.x > 30.0 && /*ball.y > 20.0 &&*/ me.x > ball.x - 27 &&
        mate_min < opp_min && mate_min < 15 && (wm.self().stamina() > 3800 || me.x > ball.x - 15) )
    {
        s_recover_mode = false;
    }

    if( wm.self().unum() == 7 && ball.x > 30.0 && /*ball.y < -20.0 &&*/ me.x > ball.x - 27 &&
        mate_min < opp_min && mate_min < 15 && (wm.self().stamina() > 3800 || me.x > ball.x - 15) )
    {
        s_recover_mode = false;
    }

    if( wm.self().unum() > 5 && wm.ourDefenseLineX() > ball.x + 1.0 )
    {
        s_recover_mode = false;
    }

    if( ball.x > -36.0 && me.x < -37.0 )
    {
        s_recover_mode = false;
    }


/// setting recover mode TRUE

    if( wm.self().unum() > 8 && wm.ball().pos().dist(wm.self().pos()) > 30.0 &&
        wm.self().stamina() < ServerParam::i().staminaMax() * 0.7 )
    {
        s_recover_mode = true;
    }





    /*--------------------------------------------------------*/
    double dash_power = ServerParam::i().maxDashPower();
    const double my_inc
        = wm.self().playerType().staminaIncMax()
        * wm.self().recovery();

    if( wm.self().unum() == 6 && isSoftMarking && ball.x < 10.0 && opp_min <= mate_min )
    {
        dash_power = ServerParam::i().maxDashPower();
    }
    else if( wm.self().unum() < 6 && isSoftMarking && ball.x < 10.0 && opp_min <= mate_min )
    {
        dash_power = ServerParam::i().maxDashPower();
    }
    if( wm.self().unum() < 6 && opp_min < mate_min && opp_min < self_min &&
        ball.x < target_point.x + 30.0 && me.x > target_point.x + 1.5 )
    {
        dash_power = ServerParam::i().maxDashPower();
    }
    else if( wm.self().unum() > 8 && mate_min > opp_min + 1 && me.x < wm.offsideLineX() + 1.0 &&
        wm.self().stamina() < ServerParam::i().staminaMax() * 0.85 &&
        ball.x < 20.0 && me.dist(target_point) < 20.0 && fabs(me.y - target_point.y) < 12.0 )
    {
        dash_power = ServerParam::i().maxDashPower() * 0.3;
    }
    else if( (wm.self().unum() == 7 || wm.self().unum() == 8) &&
             ball.x > 30.0 && mate_min < opp_min &&
             (wm.self().stamina() > 3800 || me.x > ball.x - 15) )
    {
        dash_power = ServerParam::i().maxDashPower();
    }
    else if ( wm.ourDefenseLineX() > wm.self().pos().x
         && ball.x < wm.ourDefenseLineX() + 25.0 )
    {
        dlog.addText( Logger::TEAM,
                            __FILE__": (getDashPower) correct DF line. keep max power" );
        // keep max power
        dash_power = ServerParam::i().maxDashPower();
    }
    else if ( wm.self().unum() > 5 && wm.ourDefenseLineX() > wm.self().pos().x &&
              wm.ball().pos().x < wm.ourDefenseLineX() + 25.0 )
    {
        // keep max power
        dash_power = ServerParam::i().maxDashPower();
    }
    else if ( ! wm.existKickableTeammate()
              && opp_min < self_min
              && opp_min < mate_min
              && target_point.x < wm.self().pos().x
              && target_point.x < -23.0
              && wm.ball().inertiaPoint( opp_min ).x < -23.0 )
    {
        dlog.addText( Logger::TEAM,
                            __FILE__": (getDashPower) defense dash" );
        dash_power = ServerParam::i().maxDashPower();
    }
    else if ( s_recover_mode )
    {
        dash_power = my_inc - 25.0; // preffered recover value
        if ( dash_power < 0.0 ) dash_power = 0.0;

        dlog.addText( Logger::TEAM,
                            __FILE__": (getDashPower) recovering" );
    }
    // exist kickable teammate
    else if ( wm.existKickableTeammate()
              && wm.ball().distFromSelf() < 20.0 )
    {
        dash_power = std::min( my_inc * 1.1,
                               ServerParam::i().maxDashPower() );
        dlog.addText( Logger::TEAM,
                            __FILE__": (getDashPower) exist kickable teammate. dash_power=%.1f",
                            dash_power );
    }
    // in offside area
    else if ( wm.self().pos().x > wm.offsideLineX() - 0.2 )
    {
        dash_power = ServerParam::i().maxDashPower();
        dlog.addText( Logger::TEAM,
                            __FILE__": (getDashPower) in offside area. dash_power=%.1f",
                            dash_power );
    }
    // normal
    else
    {
        dash_power = std::min( my_inc * 1.7,
                               ServerParam::i().maxDashPower() );
        dlog.addText( Logger::TEAM,
                            __FILE__": (getDashPower) normal mode dash_power=%.1f",
                            dash_power );
    }

    if( mate_min < opp_min &&
        target_point.x > wm.offsideLineX() - 8.0 &&
        wm.ball().pos().x > wm.offsideLineX() - 20.0 &&
        wm.offsideLineX() < 30.0 &&
        wm.self().unum() > 8 )
    {
        dash_power = ServerParam::i().maxDashPower();
    }

    if( wm.ball().pos().x > 33.0 && wm.ball().pos().absY() > 12.0 &&
        wm.self().pos().x > 32.0 &&
        mate_min < opp_min && mate_min < 6 && wm.self().stamina() > 3000.0 &&
        wm.self().unum() > 8 )
    {
        dash_power = ServerParam::i().maxDashPower();
    }

    if( wm.ball().pos().x > 33.0 && wm.ball().pos().absY() > 12.0 &&
        wm.self().pos().x > 20.0 &&
        mate_min < opp_min && mate_min < 6 && wm.self().stamina() > 3000.0 &&
        (wm.self().unum() == 7 || wm.self().unum() == 8) )
    {
        dash_power = ServerParam::i().maxDashPower();
    }


    return dash_power;
}
