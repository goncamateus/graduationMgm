/*

   Z----------------------------------Z
   |                                  |
   |    File Created By: Amir Tavafi  |
   |                                  |
   |    Date Created:    2009/11/30,  |
   |                     1388/08/09   |
   |                                  |
   Z----------------------------------Z

*/

#include "bhv_marlik_block.h"
#include "bhv_basic_tackle.h"

#include <rcsc/action/body_intercept.h>
#include <rcsc/action/basic_actions.h>
#include <rcsc/action/body_go_to_point.h>
#include <rcsc/action/neck_scan_field.h>
#include <rcsc/action/neck_turn_to_ball_or_scan.h>

#include <rcsc/player/player_agent.h>
#include <rcsc/player/debug_client.h>
#include <rcsc/player/soccer_intention.h>
#include <rcsc/player/intercept_table.h>

#include <rcsc/common/server_param.h>

bool Bhv_MarlikBlock::isInBlockPoint = false;
int Bhv_MarlikBlock::timeAtBlockPoint = 0;

/// Execute Block Action
bool Bhv_MarlikBlock::execute( rcsc::PlayerAgent * agent )
{
  const rcsc::WorldModel & wm = agent->world();

  rcsc::Vector2D me = wm.self().pos();
  rcsc::Vector2D ball = wm.ball().pos();
  
  int myCycles = wm.interceptTable()->selfReachCycle();
  int tmmCycles = wm.interceptTable()->teammateReachCycle();
  int oppCycles = wm.interceptTable()->opponentReachCycle();

  int num = wm.self().unum();
  
  if( ( wm.self().unum() == 2 && ball.x < -46.0 && ball.y > -18.0 && ball.y < -6.0 &&
        oppCycles <= 3 && oppCycles <= tmmCycles && ball.dist(me) < 9.0 ) ||
      ( wm.self().unum() == 3 && ball.x < -46.0 && ball.y <  18.0 && ball.y >  6.0  &&
        oppCycles <= 3 && oppCycles <= tmmCycles && ball.dist(me) < 9.0 ) )
        if( doBlockMove( agent ) )
          return true;

  rcsc::Vector2D opp = wm.opponentsFromBall().front()->pos();

  if( doInterceptBall2011( agent ) )
     return true;
  if( Bhv_BasicTackle( 0.85, 60.0 ).execute( agent ) )
     return true;

  if( (wm.self().unum() == 2 || wm.self().unum() == 3) && ball.x < -30.0 && ball.absY() > 18.0 )
     return false;

  if( num < 7 && ball.x < -30.0 && ball.x > -40.0 &&
      wm.countTeammatesIn( rcsc::Circle2D( rcsc::Vector2D(opp.x-3.0,opp.y), 3.0 ), 3, false ) > 2 )
    return false;
  
  
  if( (wm.self().unum() == 2 || wm.self().unum() == 3) && ball.x > wm.ourDefenseLineX() + 6.0 &&
      ball.x > -30.0 && ball.x < 0.0 )
     return false;

  if( wm.self().unum() == 6 && ball.x > wm.ourDefenseLineX() + 15.0 )
     return false;

  if( wm.self().unum() == 2 && ball.x < wm.ourDefenseLineX() + 7.0 && ball.absY() < 12.0 &&
      ball.y < wm.self().pos().y + 9.0 && ball.y > wm.self().pos().y - 3.0 && ball.x > -38.0 )
        if( doBlockMove( agent ) )
          return true;

  if( wm.self().unum() < 6 && ball.x < wm.ourDefenseLineX() + 5.0 &&
      ball.dist(wm.self().pos()) < 5.0 && ball.x > -38.0 ) // IO2011: < 6.5
        if( doBlockMove( agent ) )
          return true;

  if( wm.self().unum() == 3 && ball.x < wm.ourDefenseLineX() + 7.0 && ball.absY() < 12.0 &&
      ball.y > wm.self().pos().y + 7.0 && ball.y < wm.self().pos().y + 3.0 && ball.x > -38.0 )
        if( doBlockMove( agent ) )
          return true;

  if( (wm.self().unum() > 5 && wm.self().unum() <= 8 && ball.x < 25.0) ||
      ( (wm.self().unum() <=5 && ball.x < -30.0) ||
        (ball.x < wm.ourDefenseLineX() + 8.5 && ball.x > -30.0 && ball.x < -5.0) ) )
  {

    if( ball.x < -36 && ball.absY() < 7 &&
        myCycles > oppCycles && myCycles <= tmmCycles && oppCycles < tmmCycles )
    {
      if( doBlockMove( agent ) )
        return true;
    }
    if( myCycles > oppCycles && myCycles < tmmCycles + 1 && oppCycles < tmmCycles )
    {
      if( doBlockMove( agent ) )
        return true;
    }

  }

  isInBlockPoint = false;
  timeAtBlockPoint = 0;

return false;
}

/// Execute Block Action
bool Bhv_MarlikBlock::execute2010( rcsc::PlayerAgent * agent )
{
  const rcsc::WorldModel & wm = agent->world();

  rcsc::Vector2D me = wm.self().pos();
  rcsc::Vector2D ball = wm.ball().pos();

  int myCycles = wm.interceptTable()->selfReachCycle();
  int tmmCycles = wm.interceptTable()->teammateReachCycle();
  int oppCycles = wm.interceptTable()->opponentReachCycle();

  if( ( wm.self().unum() == 2 && ball.x < -46.0 && ball.y > -18.0 && ball.y < -6.0 &&
        oppCycles <= 3 && oppCycles <= tmmCycles && ball.dist(me) < 9.0 ) ||
      ( wm.self().unum() == 3 && ball.x < -46.0 && ball.y <  18.0 && ball.y >  6.0  &&
        oppCycles <= 3 && oppCycles <= tmmCycles && ball.dist(me) < 9.0 ) )
        if( doBlockMove( agent ) )
          return true;

  if( doInterceptBall( agent ) )
     return true;
  if( Bhv_BasicTackle( 0.85, 60.0 ).execute( agent ) )
     return true;

  if( (wm.self().unum() == 2 || wm.self().unum() == 3) && ball.x < -30.0 && ball.absY() > 18.0 )
     return false;

  if( (wm.self().unum() == 2 || wm.self().unum() == 3) && ball.x > wm.ourDefenseLineX() + 6.0 &&
      ball.x > -30.0 && ball.x < 0.0 )
     return false;

  if( wm.self().unum() == 6 && ball.x > wm.ourDefenseLineX() + 15.0 )
     return false;

  if( wm.self().unum() == 2 && ball.x < wm.ourDefenseLineX() + 7.0 && ball.absY() < 12.0 &&
      ball.y < wm.self().pos().y + 9.0 && ball.y > wm.self().pos().y - 3.0 && ball.x > -38.0 )
        if( doBlockMove( agent ) )
          return true;

  if( wm.self().unum() < 6 && ball.x < wm.ourDefenseLineX() + 5.0 &&
      ball.dist(wm.self().pos()) < 5.0 && ball.x > -38.0 ) // IO2011: < 6.5
        if( doBlockMove( agent ) )
          return true;

  if( wm.self().unum() == 3 && ball.x < wm.ourDefenseLineX() + 7.0 && ball.absY() < 12.0 &&
      ball.y > wm.self().pos().y + 7.0 && ball.y < wm.self().pos().y + 3.0 && ball.x > -38.0 )
        if( doBlockMove( agent ) )
          return true;

  if( (wm.self().unum() > 5 && wm.self().unum() <= 8 && ball.x < 25.0) ||
      ( (wm.self().unum() <=5 && ball.x < -30.0) ||
        (ball.x < wm.ourDefenseLineX() + 8.5 && ball.x > -30.0 && ball.x < -5.0) ) )
  {

    if( ball.x < -36 && ball.absY() < 7 &&
        myCycles > oppCycles && myCycles <= tmmCycles && oppCycles < tmmCycles )
    {
      if( doBlockMove( agent ) )
        return true;
    }
    if( myCycles > oppCycles && myCycles < tmmCycles + 1 && oppCycles < tmmCycles )
    {
      if( doBlockMove( agent ) )
        return true;
    }

  }


return false;
}

/// Do Intercept if Possible
bool Bhv_MarlikBlock::doInterceptBall( rcsc::PlayerAgent * agent )
{

    const rcsc::WorldModel & wm = agent->world();

    rcsc::Vector2D ball = wm.ball().pos();
    rcsc::Vector2D me = wm.self().pos();

    int myCycles = wm.interceptTable()->selfReachCycle();
    int tmmCycles = wm.interceptTable()->teammateReachCycle();
    int oppCycles = wm.interceptTable()->opponentReachCycle();

    if ( myCycles < oppCycles && myCycles < tmmCycles )
    {
        rcsc::Body_Intercept().execute( agent );
        agent->setNeckAction( new rcsc::Neck_TurnToBall() );
        return true;
    }

    rcsc::Vector2D myInterceptPos = wm.ball().inertiaPoint( myCycles );
    rcsc::Vector2D opp_reach_pos = wm.ball().inertiaPoint( oppCycles );

    if ( ( (myCycles <= 4 && oppCycles >= 4) || myCycles <= 2 ) &&
         me.x < myInterceptPos.x && me.x < ball.x - 0.5 &&
         ( std::fabs( ball.y - me.y ) < 1.5 || ball.absY() < me.absY() ) )
    {
        rcsc::Body_Intercept().execute( agent );
        agent->setNeckAction( new rcsc::Neck_TurnToBall() );
        return true;
    }


return false;
}

/// Do Intercept if Possible
bool Bhv_MarlikBlock::doInterceptBall2011( rcsc::PlayerAgent * agent )
{

    const rcsc::WorldModel & wm = agent->world();

    rcsc::Vector2D ball = wm.ball().pos();
    rcsc::Vector2D me = wm.self().pos();

    int myCycles = wm.interceptTable()->selfReachCycle();
    int tmmCycles = wm.interceptTable()->teammateReachCycle();
    int oppCycles = wm.interceptTable()->opponentReachCycle();

    if ( myCycles < oppCycles && myCycles < tmmCycles )
    {
        rcsc::Body_Intercept().execute( agent );
        agent->setNeckAction( new rcsc::Neck_TurnToBall() );
        return true;
    }

    rcsc::Vector2D myInterceptPos = wm.ball().inertiaPoint( myCycles );
    rcsc::Vector2D opp_reach_pos = wm.ball().inertiaPoint( oppCycles );

/*
    if ( ( (myCycles <= 4 && oppCycles >= 4) || myCycles <= 2 ) &&
         me.x < myInterceptPos.x && me.x < ball.x - 0.5 &&
         ( std::fabs( ball.y - me.y ) < 1.5 || ball.absY() < me.absY() ) )
    {
        rcsc::Body_Intercept().execute( agent );
        agent->setNeckAction( new rcsc::Neck_TurnToBall() );
        return true;
    }
*/

return false;
}

/// Going to Block Point & related Body Turns
bool Bhv_MarlikBlock::doBlockMove( rcsc::PlayerAgent * agent )
{
  const rcsc::WorldModel & wm = agent->world();

  rcsc::Vector2D ball = wm.ball().pos();
  rcsc::Vector2D me = wm.self().pos() + wm.self().vel();

  if( ball.x < -20.0 && ball.x > -37.0 && ball.absY() < 20.0 )
    return doBlockMove2011(agent);
  
  
  int myCycles = wm.interceptTable()->selfReachCycle();
  int tmmCycles = wm.interceptTable()->teammateReachCycle();
  int oppCycles = wm.interceptTable()->opponentReachCycle();

  rcsc::Vector2D blockPos = getBlockPoint( agent );
  rcsc::Vector2D opp = wm.opponentsFromBall().front()->pos() + wm.opponentsFromBall().front()->vel();
  rcsc::Vector2D goal = rcsc::Vector2D(-52.0, 0.0);

  double dashPower = getBlockDashPower( agent, blockPos );


  // opponent target
  rcsc::Vector2D oppPos = wm.opponentsFromBall().front()->pos();
  rcsc::Vector2D oppTarget = rcsc::Vector2D( -52.5, oppPos.y*0.95 ); // oppPos.y*0.975

    if( ball.x < -34.0 && ball.x > -40.0 )
         oppTarget = rcsc::Vector2D( -48.0, 0.0 );

    if( ball.x < -40.0 )
         oppTarget = rcsc::Vector2D( -48.0, 0.0 );

  rcsc::Line2D targetLine = rcsc::Line2D( oppTarget, oppPos );

  const rcsc::PlayerObject * opponent = wm.interceptTable()->fastestOpponent();
  rcsc::AngleDeg opp_body;


   if( ball.x < -36.0 && ball.absY() > 12.0 && me.dist(goal) > opp.dist(goal) )
     return false;


   int minOppDist = 3.5;

    if( ball.x < -36.0 && ball.absY() > 12.0 )
         minOppDist = 3.5;


  int minInterceptCycles = 2;

    if( ball.x < -36.0 && ball.absY() > 12.0 && me.dist(goal) < opp.dist(goal) )
         minInterceptCycles = 2; // 3 bud!
    if( ball.x < -36.0 && ball.absY() < 10.0 )
         minInterceptCycles = 2;

  static rcsc::Vector2D opp_static_pos = opp;

    if ( myCycles <= 2 )
    {
      rcsc::Body_Intercept().execute( agent );
      agent->setNeckAction( new rcsc::Neck_TurnToBallOrScan() );
      return true;
    }

    if ( myCycles < 20 && ( myCycles < tmmCycles || ( myCycles < tmmCycles + 3 && tmmCycles > 3 ) ) &&
         myCycles <= oppCycles + 1 )
    {
      rcsc::Body_Intercept().execute( agent );
      agent->setNeckAction( new rcsc::Neck_TurnToBallOrScan() );
      return true;
    }

    if ( myCycles < tmmCycles && oppCycles >= 2 && myCycles <= oppCycles + 1 )
    {
      rcsc::Body_Intercept().execute( agent );
      agent->setNeckAction( new rcsc::Neck_TurnToBallOrScan() );
      return true;
    }

   if( me.dist( blockPos ) < 1.7 && targetLine.dist(me) < 1.0 && myCycles <= minInterceptCycles &&
       targetLine.dist(opp_static_pos) < 0.7 )
   {
      rcsc::Body_Intercept().execute( agent );
      agent->setNeckAction( new rcsc::Neck_TurnToBallOrScan() );
      return true;
   }
   else if( me.dist( blockPos ) < 3.5 /* < 2.5 */&& me.dist( opp ) < minOppDist && ball.dist( opp ) < 1.3 &&
            me.dist(goal) < opp.dist(goal) && targetLine.dist(me) < 1.0 &&
            targetLine.dist(opp_static_pos) < 0.5 )
           /*&& opponent->bodyCount() <= 3 && (opp_body.abs() > 150.0 && ball.x > -36.0 ) )*/
   {
      if( !rcsc::Body_GoToPoint( opp, 0.5, dashPower, 1, false, true ).execute( agent ) )
      {
//          rcsc::AngleDeg bodyAngle = ( ball.y < me.y ? -80.0 : 80.0 );
// 
//          if( ball.x < -36.0 && ball.absY() > 13.0 )
//             bodyAngle = ( ball.x < me.x ? -180.0 : 0.0 );
// 
//          if( ball.x < -36.0 && ball.absY() < 13.0 )
//             bodyAngle = ( ball.y < me.y ? -70.0 : 70.0 );

         rcsc::AngleDeg bodyAngle = agent->world().ball().angleFromSelf();
         if ( bodyAngle.degree() < 0.0 )
                bodyAngle -= 90.0;
         else
                bodyAngle += 90.0;


         rcsc::Body_TurnToAngle( bodyAngle ).execute( agent );
      }

   }
   else if( !rcsc::Body_GoToPoint( blockPos, 0.5, dashPower, 1, false, true, 35.0 ).execute( agent ) )
   {
       rcsc::AngleDeg bodyAngle = agent->world().ball().angleFromSelf();

       if ( bodyAngle.degree() < 0.0 )
              bodyAngle -= 90.0;
       else
              bodyAngle += 90.0;

       rcsc::Body_TurnToAngle( bodyAngle ).execute( agent );
   }

   opp_static_pos = opp;


   if ( wm.ball().distFromSelf() < 20.0 &&
        ( wm.existKickableOpponent() || oppCycles <= 3 ) )
   {
       agent->setNeckAction( new rcsc::Neck_TurnToBall() );
   }
   else
   {
       agent->setNeckAction( new rcsc::Neck_TurnToBallOrScan() );
   }

return true;
}

/// Going to Block Point & related Body Turns
bool Bhv_MarlikBlock::doBlockMove2011( rcsc::PlayerAgent * agent )
{
  const rcsc::WorldModel & wm = agent->world();

  rcsc::Vector2D ball = wm.ball().pos();
  rcsc::Vector2D me = wm.self().pos() + wm.self().vel();

  int myCycles = wm.interceptTable()->selfReachCycle();
  int tmmCycles = wm.interceptTable()->teammateReachCycle();
  int oppCycles = wm.interceptTable()->opponentReachCycle();

  rcsc::Vector2D blockPos = getBlockPoint2011( agent );
  rcsc::Vector2D opp = wm.opponentsFromBall().front()->pos() + wm.opponentsFromBall().front()->vel();
  rcsc::Vector2D goal = rcsc::Vector2D(-52.0, 0.0);

  double dashPower = getBlockDashPower( agent, blockPos );


  rcsc::Vector2D oppPos = wm.opponentsFromBall().front()->pos();
  rcsc::Vector2D oppVel = wm.opponentsFromBall().front()->vel();
  
  rcsc::Vector2D oppTarget = rcsc::Vector2D( -52.5, oppPos.y*0.95 ); // oppPos.y*0.975

    if( ball.x < -34.0 && ball.x > -40.0 )
         oppTarget = rcsc::Vector2D( -53.0, 0.0 );

    if( ball.x < -42.0 )
         oppTarget = rcsc::Vector2D( -50.0, 0.0 );

  rcsc::Line2D targetLine = rcsc::Line2D( oppTarget, oppPos );

  const rcsc::PlayerObject * opponent = wm.interceptTable()->fastestOpponent();
  rcsc::AngleDeg opp_body;


   if( ball.x < -36.0 && ball.absY() > 12.0 && me.dist(goal) > opp.dist(goal) )
     return false;


   int minOppDist = 3.5;

    if( ball.x < -36.0 && ball.absY() > 12.0 )
         minOppDist = 3.5;


  int minInterceptCycles = 2;

    if( ball.x < -36.0 && ball.absY() > 12.0 && me.dist(goal) < opp.dist(goal) )
         minInterceptCycles = 2; // 3 bud!
    if( ball.x < -36.0 && ball.absY() < 10.0 )
         minInterceptCycles = 2;

  static rcsc::Vector2D opp_static_pos = opp;

    if ( myCycles <= 1 )
    {
      rcsc::Body_Intercept(true,ball).execute( agent );
      agent->setNeckAction( new rcsc::Neck_TurnToBallOrScan() );
      return true;
    }

    if ( myCycles < 20 && ( myCycles < tmmCycles || ( myCycles < tmmCycles + 2 && tmmCycles > 3 ) ) &&
         myCycles <= oppCycles )
    {
      rcsc::Body_Intercept(true,ball).execute( agent );
      agent->setNeckAction( new rcsc::Neck_TurnToBallOrScan() );
      return true;
    }

    if ( myCycles < tmmCycles && oppCycles >= 2 && myCycles <= oppCycles  )
    {
      rcsc::Body_Intercept(true,ball).execute( agent );
      agent->setNeckAction( new rcsc::Neck_TurnToBallOrScan() );
      return true;
    }

   if( me.dist( blockPos ) < 1.7 && targetLine.dist(me) < 1.0 && myCycles <= minInterceptCycles &&
       targetLine.dist(opp_static_pos) < 0.2 )
   {
      timeAtBlockPoint = 0;
      rcsc::Body_Intercept(true,ball).execute( agent );
      agent->setNeckAction( new rcsc::Neck_TurnToBallOrScan() );
      return true;
   }
   else if( me.dist( blockPos ) < 3.5 /* < 2.5 */&& me.dist( opp ) < minOppDist && ball.dist( opp ) < 1.3 &&
            me.dist(goal) < opp.dist(goal) && targetLine.dist(me) < 1.0 &&
            targetLine.dist(opp_static_pos) < 0.2 && timeAtBlockPoint > 1 ) // vele harif age tu masire targetesh nabud, nare tu in if!
   {
      if( !rcsc::Body_GoToPoint( opp, 0.4, dashPower, 1, false, true ).execute( agent ) ) // dist_thr 0.5 bud
      {
//          rcsc::AngleDeg bodyAngle = ( ball.y < me.y ? -80.0 : 80.0 );
// 
//          if( ball.x < -36.0 && ball.absY() > 13.0 )
//             bodyAngle = ( ball.x < me.x ? -180.0 : 0.0 );
// 
//          if( ball.x < -36.0 && ball.absY() < 13.0 )
//             bodyAngle = ( ball.y < me.y ? -70.0 : 70.0 );

         rcsc::AngleDeg bodyAngle = agent->world().ball().angleFromSelf();
         if ( bodyAngle.degree() < 0.0 )
                bodyAngle -= 90.0;
         else
                bodyAngle += 90.0;


         rcsc::Body_TurnToAngle( bodyAngle ).execute( agent );
      }

   }
   else if( !rcsc::Body_GoToPoint( blockPos, 0.4, dashPower, 1, 100, true, 30.0 ).execute( agent ) ) // dist_thr 0.5 bud
   {
       rcsc::AngleDeg bodyAngle = agent->world().ball().angleFromSelf();

       if ( bodyAngle.degree() < 0.0 )
              bodyAngle -= 90.0;
       else
              bodyAngle += 90.0;

       rcsc::Body_TurnToAngle( bodyAngle ).execute( agent );
       timeAtBlockPoint++;
   }
   else
     timeAtBlockPoint = 0;

   opp_static_pos = opp;


   if ( wm.ball().distFromSelf() < 20.0 &&
        ( wm.existKickableOpponent() || oppCycles <= 3 ) )
   {
       agent->setNeckAction( new rcsc::Neck_TurnToBall() );
   }
   else
   {
       agent->setNeckAction( new rcsc::Neck_TurnToBallOrScan() );
   }

return true;
}

/// get Opponent BlockPoint for Moving to
rcsc::Vector2D Bhv_MarlikBlock::getBlockPoint( rcsc::PlayerAgent * agent )
{
  const rcsc::WorldModel & wm = agent->world();

  rcsc::Vector2D ball = wm.ball().pos();
  rcsc::Vector2D me = wm.self().pos();
  rcsc::Vector2D blockPos = ball;

  if( ball.x < -20.0 && ball.x > -37.0 && ball.absY() < 20.0 )
    return getBlockPoint2011(agent);
  
//   int myCycles = wm.interceptTable()->selfReachCycle();
//   int tmmCycles = wm.interceptTable()->teammateReachCycle();
  int oppCycles = wm.interceptTable()->opponentReachCycle();

  float oppDribbleSpeed = 0.675; // 0.75 bud!

  rcsc::Vector2D ballPred = ball;

   if( !wm.existKickableOpponent() )
        ballPred = wm.ball().inertiaPoint( oppCycles );

  rcsc::Vector2D oppPos = wm.opponentsFromBall().front()->pos();

//   const rcsc::PlayerObject * opponent = wm.interceptTable()->fastestOpponent();

  rcsc::Vector2D oppTarget = rcsc::Vector2D( -52.5, oppPos.y*0.95 ); // oppPos.y*0.975


   if( wm.self().unum() == 7 || wm.self().unum() == 9 )
        oppTarget = rcsc::Vector2D( -36.0, -12.0 );
   if( wm.self().unum() == 8 || wm.self().unum() == 10 )
        oppTarget = rcsc::Vector2D( -36.0, 12.0 );

   if( wm.self().unum() == 11 )
        oppTarget = rcsc::Vector2D( -30.0, 0.0 );


   if( ball.x < -34.0 && ball.x > -40.0 )
        oppTarget = rcsc::Vector2D( -48.0, 0.0 );

   if( ball.x < -40.0 )
        oppTarget = rcsc::Vector2D( -48.0, 0.0 );


   rcsc::Vector2D nextBallPos = ballPred;

  int oppTime;
   if( wm.existKickableOpponent() )
      oppTime = -1;
   else
      oppTime = oppCycles - 1;

  while( wm.self().playerType().cyclesToReachDistance( me.dist(nextBallPos) ) > oppTime && oppTime < 30 )
  {
       nextBallPos += rcsc::Vector2D::polar2vector( oppDribbleSpeed, (oppTarget - ball).dir() );
       oppTime++;
  }

  if( oppTime >= 30 )
       blockPos = rcsc::Vector2D( -48.0, ball.y );
  else
       blockPos = nextBallPos;

  if( wm.self().unum() < 6 && blockPos.x > 0.0 )
       blockPos.x = 0.0;

return blockPos;
}

/// get Opponent BlockPoint for Moving to
rcsc::Vector2D Bhv_MarlikBlock::getBlockPoint2011( rcsc::PlayerAgent * agent )
{
  const rcsc::WorldModel & wm = agent->world();

  rcsc::Vector2D ball = wm.ball().pos();
  rcsc::Vector2D me = wm.self().pos();
  rcsc::Vector2D blockPos = ball;

//   int myCycles = wm.interceptTable()->selfReachCycle();
//   int tmmCycles = wm.interceptTable()->teammateReachCycle();
  int oppCycles = wm.interceptTable()->opponentReachCycle();

  float oppDribbleSpeed = 0.675; // 0.75 bud!

  rcsc::Vector2D ballPred = ball;

   if( !wm.existKickableOpponent() )
        ballPred = wm.ball().inertiaPoint( oppCycles );

  rcsc::Vector2D oppPos = wm.opponentsFromBall().front()->pos();

//   const rcsc::PlayerObject * opponent = wm.interceptTable()->fastestOpponent();

  rcsc::Vector2D oppTarget = rcsc::Vector2D( -52.5, oppPos.y*0.95 ); // oppPos.y*0.975


   if( ball.x > 36.0 && (wm.self().unum() == 7 || wm.self().unum() == 9) )
        oppTarget = rcsc::Vector2D( -36.0, -12.0 );
   if( wm.self().unum() == 8 || wm.self().unum() == 10 )
        oppTarget = rcsc::Vector2D( -36.0, 12.0 );

   if( ball.x > 30.0 && wm.self().unum() == 11 )
        oppTarget = rcsc::Vector2D( -30.0, 0.0 );


   if( ball.x < -34.0 && ball.x > -40.0 )
        oppTarget = rcsc::Vector2D( -48.0, 0.0 );

   if( ball.x < -40.0 )
        oppTarget = rcsc::Vector2D( -48.0, 0.0 );


   rcsc::Vector2D nextBallPos = ballPred;

  int oppTime;
   if( wm.existKickableOpponent() )
      oppTime = -1;
   else
      oppTime = oppCycles - 1;

  while( wm.self().playerType().cyclesToReachDistance( me.dist(nextBallPos) ) > oppTime && oppTime < 50 )
  {
       nextBallPos += rcsc::Vector2D::polar2vector( oppDribbleSpeed, (oppTarget - ball).dir() );
       oppTime++;
  }

  if( oppTime >= 30 )
       blockPos = rcsc::Vector2D( -48.0, ball.y );
  else
       blockPos = nextBallPos;

  if( wm.self().unum() < 6 && blockPos.x > 0.0 )
       blockPos.x = 0.0;

return blockPos;
}

/// get Opponent BlockPoint for Moving to
rcsc::Vector2D Bhv_MarlikBlock::getBlockPoint2011_backUp( rcsc::PlayerAgent * agent )
{

  const rcsc::WorldModel & wm = agent->world();

  rcsc::Vector2D ball = wm.ball().pos();
  rcsc::Vector2D me = wm.self().pos();
  rcsc::Vector2D blockPos = ball;

//   int myCycles = wm.interceptTable()->selfReachCycle();
//   int tmmCycles = wm.interceptTable()->teammateReachCycle();
  int oppCycles = wm.interceptTable()->opponentReachCycle();

  float oppDribbleSpeed = 0.750; // 0.75 bud!

  rcsc::Vector2D ballPred = ball;

   if( !wm.existKickableOpponent() )
        ballPred = wm.ball().inertiaPoint( oppCycles );

  rcsc::Vector2D oppPos = wm.opponentsFromBall().front()->pos() +
                          wm.opponentsFromBall().front()->vel();

//   const rcsc::PlayerObject * opponent = wm.interceptTable()->fastestOpponent();

  rcsc::Vector2D oppTarget = rcsc::Vector2D( -52.5, oppPos.y*0.95 ); // oppPos.y*0.975

   if( ball.x > 36.0 && (wm.self().unum() == 7 || wm.self().unum() == 9) )
        oppTarget = rcsc::Vector2D( -36.0, -12.0 );
   if( wm.self().unum() == 8 || wm.self().unum() == 10 )
        oppTarget = rcsc::Vector2D( -36.0, 12.0 );

   if( ball.x > 30.0 && wm.self().unum() == 11 )
        oppTarget = rcsc::Vector2D( -30.0, 0.0 );


   if( ball.x < -34.0 && ball.x > -40.0 )
        oppTarget = rcsc::Vector2D( -50.0, 0.0 );

   if( ball.x < -40.0 )
        oppTarget = rcsc::Vector2D( -49.0, 0.0 );


   rcsc::Vector2D nextBallPos = ballPred;

  int oppTime = oppCycles;

  
  
  while( wm.self().playerType().cyclesToReachDistance( me.dist(nextBallPos) ) > oppTime && oppTime < 50 )
  {
    if( oppCycles > 0 )
       nextBallPos += rcsc::Vector2D::polar2vector( wm.ball().vel().r(), wm.ball().vel().dir() );         
    else
       nextBallPos += rcsc::Vector2D::polar2vector( oppDribbleSpeed, (oppTarget - ball).dir() );

    oppTime++;
    
  }
  

  if( oppTime >= 50 )
       blockPos = rcsc::Vector2D( -48.0, ball.y );
  else
       blockPos = nextBallPos;

  if( wm.self().unum() < 6 && blockPos.x > 0.0 )
       blockPos.x = 0.0;
  
  if( me.dist(blockPos) < 3 && oppCycles < 2 && oppPos.dist(ball) < 1.7 )
    blockPos = rcsc::Vector2D::polar2vector( 2.5, (oppTarget-oppPos).dir() );

return blockPos;
}

/// get Dash Power for Move
double Bhv_MarlikBlock::getBlockDashPower( const rcsc::PlayerAgent * agent,
                             const rcsc::Vector2D & blockPos )
{

    const rcsc::WorldModel & wm = agent->world();

    double dashPower = rcsc::ServerParam::i().maxDashPower();
//     const double my_inc = wm.self().playerType().staminaIncMax() * wm.self().recovery();

    int myCycles = wm.interceptTable()->selfReachCycle();
    int tmmCycles = wm.interceptTable()->teammateReachCycle();
    int oppCycles = wm.interceptTable()->opponentReachCycle();

    if ( wm.ourDefenseLineX() > wm.self().pos().x
         && wm.ball().pos().x < wm.ourDefenseLineX() + 20.0 )
    {
        dashPower = rcsc::ServerParam::i().maxDashPower();
    }
    else if ( !wm.existKickableTeammate()
              && oppCycles < myCycles
              && oppCycles < tmmCycles
              && blockPos.x < wm.self().pos().x
              && blockPos.x < -25.0
              && wm.ball().inertiaPoint( oppCycles ).x < -25.0 )
    {
        dashPower = rcsc::ServerParam::i().maxDashPower();
    }
    else
    {
        dashPower = rcsc::ServerParam::i().maxDashPower();
    }

return dashPower;
}

